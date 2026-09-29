"""CPU Toy A/B end-to-end test.

Toy는 LRT에 유리하게 만든 synthetic fixture이므로 **architecture plumbing sanity**일 뿐이며
real-data evidence가 아닙니다.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from aimo.config import LRTConfig
from aimo.lrt import fit_macro_norm_stats
from aimo.lrt_experiment import (
    Rank4LinearTransport,
    ZeroTransport,
    evaluate_transport,
    make_toy_rank4,
    make_toy_state_dependent,
    split_pairs_by_original,
    to_macro_pairs,
    train_lrt,
)

FLOOR = 1e-6


def compact_toy_b():
    """작고 빠른 Toy B fixture (landmark 2, hidden 6)."""
    bundle = make_toy_state_dependent(
        n_originals=6, n_variants=2, n_macro=8, n_landmarks=2, hidden=6, seed=0
    )
    return to_macro_pairs(bundle["pairs"], 8)


def test_toy_a_is_a_fixed_rank4_relation():
    """Toy A의 relation은 macro/landmark에 무관한 고정 ΔU = B z입니다."""
    bundle = make_toy_rank4(n_originals=4, n_variants=1, n_macro=8, seed=0)
    original, variant = bundle["pairs"][0]
    macro_o, macro_v = to_macro_pairs([(original, variant)], 8)[0]
    delta = macro_v.updates - macro_o.updates  # [G, P, 2, H]
    reference = delta[0, 0]
    for g in range(8):
        for p in range(delta.shape[1]):
            assert pytest.approx(0.0, abs=1e-5) == float((delta[g, p] - reference).abs().max())


def test_toy_b_relation_is_state_dependent():
    """Toy B는 같은 pair 안에서도 macro/landmark에 따라 relation이 달라집니다."""
    pairs = compact_toy_b()
    macro_o, macro_v = pairs[0]
    delta = macro_v.updates - macro_o.updates
    spread = float((delta - delta[0, 0]).abs().max())
    assert spread > 1e-3


@pytest.mark.slow
def test_lrt_beats_zero_on_toy_b_without_behavior_labels():
    """Toy B에서 transport ratio < 1 이고 correct support가 swapped support보다 좋습니다."""
    pairs = compact_toy_b()
    train_pairs, validation_pairs = split_pairs_by_original(pairs, holdout=2)
    stats = fit_macro_norm_stats([o for o, _ in train_pairs])
    cfg = replace(LRTConfig(), denominator_floor=FLOOR)
    result = train_lrt(
        train_pairs, validation_pairs, stats, cfg, floor=FLOOR, max_epochs=90,
        patience=90, seed=0,
    )
    zero = evaluate_transport(
        None, validation_pairs, stats, floor=FLOOR, baseline=ZeroTransport()
    )
    learned = evaluate_transport(
        result["model"], validation_pairs, stats, floor=FLOOR, support_swap=True, eval_seed=0
    )
    assert zero["transport_ratio"] == pytest.approx(1.0, rel=1e-5)
    # Zero baseline보다 낮은 transport ratio.
    assert learned["transport_ratio"] < 0.9
    # correct relation code가 support-swap보다 좋아야 합니다 (causal evidence는 아닙니다).
    assert learned["swap_gap"] > 0.0
    assert learned["swap_transport_ratio"] > learned["transport_ratio"]
    # behavior label 없이 학습했습니다.
    assert result["params"]["total"] > 0
    final_token = evaluate_transport(
        result["model"], validation_pairs, stats, floor=FLOOR,
        landmark_mode="final_token", support_swap=True, eval_seed=0,
    )
    assert final_token["landmark_mode"] == "final_token"
    assert final_token["transport_ratio"] < 1.5


def test_train_mean_and_rank4_baselines_run_on_toy_b():
    from aimo.lrt_experiment import TrainMeanTransport

    pairs = compact_toy_b()
    train_pairs, validation_pairs = split_pairs_by_original(pairs, holdout=2)
    stats = fit_macro_norm_stats([o for o, _ in train_pairs])
    for baseline in (
        TrainMeanTransport.fit(train_pairs, stats),
        Rank4LinearTransport.fit(train_pairs, stats),
    ):
        result = evaluate_transport(
            None, validation_pairs, stats, floor=FLOOR, baseline=baseline
        )
        assert result["transport_ratio"] >= 0.0
        assert result["n_originals"] == 2


# --------------------------------------------------------------------------------------
# CLI / audit
# --------------------------------------------------------------------------------------


def test_macro_audit_reports_fine_vs_macro(tmp_path):
    from aimo.lrt_experiment import audit_macro_page

    bundle = make_toy_state_dependent(n_originals=3, n_variants=1, n_macro=8, seed=0)
    report = audit_macro_page(bundle["pairs"], 8)
    assert report["n_pairs"] == 3
    assert report["max_macro_residual_identity_error"] < 1e-3
    assert report["mean_macro_over_fine_energy"] > 0.0
    assert report["all_common_landmark_count_mean"] > report["final_token_landmark_count_mean"]
    assert any("diagnostic" in note for note in report["notes"])
    for row in report["pairs"]:
        assert len(row["boundaries"]) == 9
        assert row["fine_relation_energy"] > 0.0


def test_macro_audit_rejects_a_page_store_shallower_than_g():
    from aimo.lrt_experiment import audit_macro_page

    bundle = make_toy_state_dependent(
        n_originals=2, n_variants=1, n_blocks=8, n_macro=4, seed=0
    )
    with pytest.raises(ValueError, match="n_macro <= L"):
        audit_macro_page(bundle["pairs"], 16)


def test_audit_and_lrt_commands_run(tmp_path, capsys):
    from aimo.cli import main

    assert main(["audit-macro-page", "--run-dir", str(tmp_path / "audit")]) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["source"] == "cpu_toy_fixture"
    assert (tmp_path / "audit" / "macro_page_audit.json").exists()

    assert main(["lrt-experiment", "--run-dir", str(tmp_path / "spec")]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["schema"] == "aimo-lrt-v1"
    assert spec["status"] == "SPEC_ONLY / SERVER-UNTESTED"
    assert spec["lrt_config"]["denominator_floor"] is None
    assert spec["folds"] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert (tmp_path / "spec" / "lrt_spec.json").exists()


def test_lrt_toy_command_requires_an_explicit_floor(tmp_path):
    from aimo.cli import main

    with pytest.raises(ValueError, match="denominator_floor is null"):
        main(["lrt-experiment", "--run-dir", str(tmp_path / "toy"), "--toy", "B"])
