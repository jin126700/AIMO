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


# --------------------------------------------------------------------------------------
# DeepMath adapter (v2 primary 데이터 경로)
# --------------------------------------------------------------------------------------


def _deepmath_fixture(tmp_path):
    """작은 schema fixture. 실제 DeepMath row가 아니라 형식 확인용입니다."""
    import json

    rows = [
        {
            "row_id": "d1",
            "question": "Let n be the least positive integer with 7 | 3^n - 1. Find n.",
            "final_answer": "6",
            "difficulty": 5.5,
            "topic": "Mathematics -> Number Theory -> Congruences",
            "r1_solution_1": "long chain of thought",
        },
        {
            "row_id": "d2",
            "question": "As shown in the figure, ABC is a triangle. Find its area.",
            "final_answer": "12",
            "difficulty": 3.0,
            "topic": "Mathematics -> Geometry -> Plane Geometry",
        },
        {
            "row_id": "d3",
            "question": "Count the subsets of {1,...,10} whose elements sum to exactly 20.",
            "final_answer": "51",
            "difficulty": 7.5,
            "topic": "Mathematics -> Discrete Mathematics -> Combinatorics",
        },
        {
            "row_id": "d4",
            "question": "Evaluate the limit of (1 + 1/n)^n as n grows without bound.",
            "final_answer": "e",
            "difficulty": 2.0,
            "topic": "Mathematics -> Calculus -> Limits",
        },
        {
            "row_id": "d5",
            "question": "Let f(x) = x^3 - 3x + 1. Find the sum of all real roots of f.",
            "final_answer": "0",
            "difficulty": 6.0,
            "topic": "Mathematics -> Algebra -> Polynomials",
        },
        {
            "row_id": "d6",
            "question": "Find the number of ordered pairs (a, b) of positive integers"
            " with ab = 360.",
            "final_answer": "24",
            "difficulty": 4.0,
            "topic": "Mathematics -> Number Theory -> Factorization",
        },
    ]
    path = tmp_path / "deepmath_sample.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_deepmath_probe_is_server_pending():
    from aimo.adapters.deepmath import DATASET_ID, probe_deepmath

    probe = probe_deepmath()
    assert probe["status"] == "SERVER_PENDING"
    assert probe["dataset_id"] == DATASET_ID == "zwhe99/DeepMath-103K"


def test_deepmath_reads_only_the_fields_it_needs(tmp_path):
    from aimo.adapters.deepmath import EXCLUDED_FROM_MODEL, load_local_rows

    rows = load_local_rows(_deepmath_fixture(tmp_path), revision="fixture")
    assert len(rows) == 6
    assert rows[0].n_r1_solutions == 1
    # r1 풀이 내용은 보관하지 않습니다.
    assert not hasattr(rows[0], "r1_solution_1")
    assert "r1_solution_1" in EXCLUDED_FROM_MODEL
    metadata = rows[0].as_metadata()
    assert metadata["difficulty_scale"] == "deepmath_native"  # MATH Level과 다른 척도
    assert metadata["source_revision"] == "fixture"


def test_deepmath_missing_required_field_fails():
    from aimo.adapters.deepmath import validate_row

    with pytest.raises(ValueError, match="missing required field"):
        validate_row({"question": "x"}, 0)


def test_deepmath_candidate_selection_rejects_figures_and_off_topic(tmp_path):
    from aimo.adapters.deepmath import load_local_rows, select_candidates

    rows = load_local_rows(_deepmath_fixture(tmp_path))
    report = select_candidates(rows, max_originals=10)
    ids = [row.row_id for row in report.candidates]
    assert "d2" not in ids  # figure 의존
    assert "d4" not in ids  # 우선 topic 범위 밖
    assert {"d1", "d3", "d5", "d6"} == set(ids)
    assert report.rejected["figure_or_external_reference"] == 1
    assert "후보 수" in report.as_dict()["note"]


def test_deepmath_candidate_cap_is_configurable(tmp_path):
    from aimo.adapters.deepmath import (
        DEFAULT_CANDIDATE_ORIGINALS,
        load_local_rows,
        select_candidates,
    )

    assert DEFAULT_CANDIDATE_ORIGINALS == 300
    rows = load_local_rows(_deepmath_fixture(tmp_path))
    report = select_candidates(rows, max_originals=2)
    assert len(report.candidates) == 2
    assert report.rejected["over_candidate_cap"] >= 1


def test_deepmath_splits_keep_each_original_in_one_split(tmp_path):
    from aimo.adapters.deepmath import (
        SPLIT_NAMES,
        assign_splits,
        load_local_rows,
        select_candidates,
    )

    rows = load_local_rows(_deepmath_fixture(tmp_path)) * 6  # 후보 수를 늘립니다
    for index, row in enumerate(rows):
        row.row_id = f"{row.row_id}-{index}"
    report = select_candidates(rows, max_originals=24)
    splits = assign_splits(report.candidates, seed=0)
    assert set(splits) == set(SPLIT_NAMES)
    seen: set[str] = set()
    for members in splits.values():
        ids = {row.row_id for row in members}
        assert not (ids & seen)  # 같은 original이 두 split에 들어가지 않습니다
        seen |= ids
    assert seen == {row.row_id for row in report.candidates}
    # harder split은 native difficulty 상위에서 고릅니다.
    if splits["harder"]:
        assert min(row.difficulty for row in splits["harder"]) >= 5.0


def test_deepmath_overlap_check_does_not_claim_decontamination(tmp_path):
    from aimo.adapters.deepmath import benchmark_overlap, load_local_rows

    rows = load_local_rows(_deepmath_fixture(tmp_path))
    report = benchmark_overlap(rows, [rows[4].question])
    assert report["exact_matches"] == ["d5"]
    assert report["decontamination_verified"] is False


# --------------------------------------------------------------------------------------
# Perturbation adapter
# --------------------------------------------------------------------------------------


def _pair_base():
    from aimo.adapters.perturbation import PairCandidate

    return PairCandidate(
        pair_id="p0",
        original_id="o1",
        variant_id="o1#v0",
        panel_id="o1#panel",
        original_text="Let x be a positive integer with \\dfrac{x}{2} + 3 = 7.  What is x?",
        variant_text="",
        original_answer="8",
        variant_answer="8",
        source_dataset="deepmath",
    )


def test_only_allowlisted_formatting_rules_are_applied():
    from aimo.adapters import AdapterUnavailable
    from aimo.adapters.perturbation import apply_formatting, make_formatting_variant

    variant = make_formatting_variant(_pair_base(), ["dfrac_to_frac", "collapse_spaces"])
    assert "\\frac" in variant.variant_text and "\\dfrac" not in variant.variant_text
    assert variant.semantic_valid == "verified"
    with pytest.raises(AdapterUnavailable, match="allowlist"):
        apply_formatting("x", ["shuffle_sentences"])


def test_alpha_rename_scope_checks():
    from aimo.adapters.perturbation import make_alpha_rename_variant

    ok = make_alpha_rename_variant(_pair_base(), "x", "y")
    assert ok.semantic_valid == "verified" and not ok.rejection_reasons
    assert " y " in ok.variant_text or "y" in ok.variant_text
    reserved = make_alpha_rename_variant(_pair_base(), "x", "e")
    assert reserved.rejection_reasons == ["reserved_math_symbol"]
    collision = make_alpha_rename_variant(_pair_base(), "x", "a")
    assert collision.rejection_reasons == ["new_name_collides_with_existing_token"]
    unit = make_alpha_rename_variant(_pair_base(), "x", "m")
    assert "unit_token" in unit.rejection_reasons
    for candidate in (reserved, collision, unit):
        assert candidate.semantic_valid == "unknown"  # 지원하지 않으면 검증 대기로 남깁니다


def test_import_pairs_requires_semantic_evidence(tmp_path):
    import json

    from aimo.adapters.perturbation import import_verified_pairs

    rows = [
        {
            "original_id": "o1", "variant_id": "o1#v0", "original_text": "A",
            "variant_text": "B", "original_answer": "1", "variant_answer": "1",
            "semantic_validation_evidence": ["human_check#42"], "semantic_valid": "verified",
            "source_revision": "rev1",
        },
        {
            # 같은 정답이지만 evidence가 없으므로 verified가 되지 않습니다.
            "original_id": "o1", "variant_id": "o1#v1", "original_text": "A",
            "variant_text": "C", "original_answer": "1", "variant_answer": "1",
            "semantic_validation_evidence": [], "source_revision": "rev1",
        },
        {
            "original_id": "o2", "variant_id": "o2#v0", "original_text": "D",
            "variant_text": "E", "original_answer": "2", "variant_answer": "5",
            "answer_mapping": "answer scaled by 2.5",
            "semantic_validation_evidence": ["human_check#7"], "semantic_valid": "verified",
            "source_revision": "rev1", "namespace": "stress",
        },
    ]
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    candidates, report = import_verified_pairs(path)
    assert report["n_verified"] == 2 and report["n_unknown"] == 1
    usable = {c.variant_id: c.usable_for_behavior for c in candidates}
    assert usable == {"o1#v0": True, "o1#v1": False, "o2#v0": False}
    # stress namespace는 meaning-preserving primary와 분리됩니다.
    stress = next(c for c in candidates if c.variant_id == "o2#v0")
    assert stress.namespace == "stress"
    assert stress.answer_mapping == "answer scaled by 2.5"
    assert all(c.usability == "research_only" for c in candidates)


def test_frozen_pairs_are_not_silently_replaced(tmp_path):
    from aimo.adapters import AdapterUnavailable
    from aimo.adapters.perturbation import FrozenPairStore

    store = FrozenPairStore.freeze([_pair_base()])
    path = tmp_path / "frozen.json"
    store.save(path)
    with pytest.raises(AdapterUnavailable, match="not silently replaced"):
        store.save(path)
    reloaded = FrozenPairStore.load(path)
    assert reloaded.frozen_hash == store.frozen_hash
    assert reloaded.report()["n_candidates"] == 1


# --------------------------------------------------------------------------------------
# research thinking protocol
# --------------------------------------------------------------------------------------


def test_thinking_profile_defaults_and_calibration():
    from aimo.adapters.qwen import thinking_ready
    from aimo.config import ThinkingProfileConfig

    profile = ThinkingProfileConfig()
    assert profile.model_id == "Qwen/Qwen3-4B"
    assert profile.enable_thinking and profile.do_sample
    assert (profile.temperature, profile.top_p, profile.top_k, profile.min_p) == (
        0.6, 0.95, 20, 0.0
    )
    report = thinking_ready(profile)
    assert report["status"] == "SERVER_PENDING"
    assert "max_new_tokens" in report["needs_calibration"]
    assert "samples_per_prompt" in report["needs_calibration"]
    assert "공식 AIMO" in report["note"]


def test_mid_thinking_number_is_not_scored_as_correct():
    from aimo.adapters.qwen import classify_thinking_slot

    text = "<think>maybe the answer is 42</think> I could not finish."
    assert classify_thinking_slot(
        started=True, infra_error=False, hit_cap=False, is_final_cap=False, text=text, gold="42"
    ) == "U_score"


def test_unclosed_thinking_is_cap_hit_not_wrong():
    from aimo.adapters.qwen import classify_thinking_slot

    assert classify_thinking_slot(
        started=True, infra_error=False, hit_cap=True, is_final_cap=False,
        text="<think>still working, maybe 42", gold="42",
    ) == "X"


def test_cap_hit_and_final_cap_are_distinguished():
    from aimo.adapters.qwen import classify_thinking_slot

    text = "<think>done</think> Final answer: 42"
    assert classify_thinking_slot(
        started=True, infra_error=False, hit_cap=True, is_final_cap=False, text=text, gold="42"
    ) == "X"
    assert classify_thinking_slot(
        started=True, infra_error=False, hit_cap=True, is_final_cap=True, text=text, gold="42"
    ) == "C"


def test_total_context_is_checked_not_truncated():
    from aimo.adapters.qwen import check_total_context

    check_total_context(100, 200, 400)
    with pytest.raises(ValueError, match="exceeds the configured limit"):
        check_total_context(300, 200, 400)
    with pytest.raises(ValueError, match="not calibrated"):
        check_total_context(1, 1, None)


def test_rope_scaling_change_makes_a_different_protocol_hash():
    from aimo.config import ThinkingProfileConfig

    base = ThinkingProfileConfig()
    scaled = ThinkingProfileConfig(rope_scaling="yarn-2x")
    assert base.protocol_hash() != scaled.protocol_hash()
