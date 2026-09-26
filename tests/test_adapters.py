"""서버 adapter test. transformers가 없으면 해당 test는 skip합니다."""

from __future__ import annotations

import pytest
import torch

from aimo.adapters import AdapterUnavailable
from aimo.adapters.mathgap import (
    OUTCOME_CAP_HIT,
    OUTCOME_CORRECT,
    OUTCOME_INFRA_ERROR,
    OUTCOME_NOT_STARTED,
    OUTCOME_UNSCORED,
    OUTCOME_WRONG,
    MathGapAdapter,
    PromptGrade,
    classify_slot,
    exact_match,
    pair_eligible,
    parse_final_answer,
    probe_mathgap,
)
from aimo.config import MathGapConfig

transformers = pytest.importorskip("transformers", reason="optional server dependency")


# --------------------------------------------------------------------------------------
# MathGAP adapter: 확인하지 못한 API는 fail-fast
# --------------------------------------------------------------------------------------


def test_mathgap_adapter_fails_fast_when_not_configured():
    with pytest.raises(AdapterUnavailable, match="SERVER_PENDING"):
        MathGapAdapter.from_config(MathGapConfig())


def test_mathgap_probe_reports_server_pending():
    probe = probe_mathgap()
    assert probe["status"] == "SERVER_PENDING"


def test_mathgap_adapter_uses_a_configured_dotted_path():
    cfg = MathGapConfig(
        revision="local-test",
        generator_path="math.fsum",
        renderer_path="math.fsum",
        oracle_path="math.fsum",
    )
    adapter = MathGapAdapter.from_config(cfg)
    assert adapter.revision == "local-test"
    assert callable(adapter.generator)


# --------------------------------------------------------------------------------------
# 고정 parser와 screening outcome
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"so the answer is \boxed{42}", "42"),
        ("Final answer: 1,024", "1024"),
        ("... 7 apples and 5 pears, total 12", "12"),
        ("no numbers here", None),
    ],
)
def test_parse_final_answer(text, expected):
    assert parse_final_answer(text) == expected


def test_exact_match_normalizes_numbers():
    assert exact_match("42", "42.0")
    assert not exact_match("41", "42")
    assert not exact_match(None, "42")


def test_cap_hit_is_x_and_not_merged_into_wrong():
    assert classify_slot(started=True, infra_error=False, hit_cap=True, text="", gold="42") == (
        OUTCOME_CAP_HIT
    )
    # 답을 냈지만 cap에 걸린 경우도 X로 남기고 C/W로 세지 않습니다.
    assert classify_slot(
        started=True, infra_error=False, hit_cap=True, text=r"\boxed{42}", gold="42"
    ) == OUTCOME_CAP_HIT
    assert classify_slot(
        started=True, infra_error=False, hit_cap=False, text=r"\boxed{41}", gold="42"
    ) == OUTCOME_WRONG


def test_outcome_codes_are_separated():
    assert classify_slot(started=False, infra_error=False, hit_cap=False, text=None, gold="1") == (
        OUTCOME_NOT_STARTED
    )
    assert classify_slot(started=True, infra_error=True, hit_cap=False, text=None, gold="1") == (
        OUTCOME_INFRA_ERROR
    )
    assert classify_slot(
        started=True, infra_error=False, hit_cap=False, text="I give up", gold="1"
    ) == OUTCOME_UNSCORED


def test_c4_requires_four_correct_slots():
    assert PromptGrade([OUTCOME_CORRECT] * 4).is_c4
    assert not PromptGrade([OUTCOME_CORRECT] * 3).is_c4
    assert not PromptGrade([OUTCOME_CORRECT] * 3 + [OUTCOME_CAP_HIT]).is_c4
    counts = PromptGrade([OUTCOME_CORRECT, OUTCOME_WRONG, OUTCOME_CAP_HIT, OUTCOME_UNSCORED]).counts
    assert counts[OUTCOME_CAP_HIT] == 1 and counts[OUTCOME_WRONG] == 1


def test_eligibility_needs_semantic_validity_and_both_c4():
    good = PromptGrade([OUTCOME_CORRECT] * 4)
    bad = PromptGrade([OUTCOME_CORRECT] * 3 + [OUTCOME_WRONG])
    assert pair_eligible(True, good, good) == (True, "eligible")
    assert pair_eligible(False, good, good) == (False, "semantic_invalid")
    assert pair_eligible(True, bad, good) == (False, "original_not_c4")
    assert pair_eligible(True, good, bad) == (False, "variant_not_c4")


# --------------------------------------------------------------------------------------
# Qwen adapter: random-init tiny config CPU 검증
# --------------------------------------------------------------------------------------


def test_probe_qwen_reports_the_available_family():
    from aimo.adapters.qwen import probe_qwen

    probe = probe_qwen()
    assert probe["available"] is True
    assert set(probe["families"]) == {"qwen3", "qwen2"}
    if not probe["qwen3_ready"]:
        assert probe["status"] == "SERVER_PENDING"


def test_select_landmarks_marks_duplicates_and_padding_invalid():
    from aimo.adapters.qwen import select_landmarks
    from aimo.page import N_LANDMARKS

    offsets, valid, rel = select_landmarks([3, 3, 5], prompt_len=20)
    assert offsets.shape == valid.shape == rel.shape == (N_LANDMARKS,)
    assert valid.tolist()[:3] == [True, False, True]  # 중복 landmark는 padding
    assert not any(valid.tolist()[3:16])  # 모자란 자리도 padding
    assert valid[-1]  # canonical final prompt token은 항상 유효
    assert int(offsets[-1]) == 19
    assert float(rel[-1]) == pytest.approx(1.0)


def test_extract_page_from_tiny_random_qwen_satisfies_the_residual_identity():
    from aimo.adapters.qwen import build_tiny_qwen, extract_page, select_landmarks

    tiny = build_tiny_qwen(hidden_size=16, n_layers=3, n_heads=2)
    torch.manual_seed(0)
    prompt_len = 30
    input_ids = torch.randint(0, 64, (1, prompt_len))
    offsets, valid, rel = select_landmarks(list(range(1, prompt_len, 2)), prompt_len)
    page = extract_page(tiny.model, input_ids, offsets, valid, rel, "o", "o#v0")
    assert tuple(page.state.shape) == (4, 17, 16)
    assert tuple(page.updates.shape) == (3, 17, 2, 16)
    assert page.residual_identity_error() < 1e-4
    assert page.provenance["n_layers"] == 3
    assert page.provenance["prompt_len"] == prompt_len


def test_screening_stays_server_pending():
    from aimo.adapters.qwen import screening_ready

    assert screening_ready()["status"] == "SERVER_PENDING"
