"""Label schema test: counts, bounds, loss routing, missing vs 0, policy mismatch."""

from __future__ import annotations

import pytest
import torch

from aimo.data import collate_panels
from aimo.labels import (
    OUTCOME_CAP_HIT,
    OUTCOME_CORRECT,
    OUTCOME_INFRA_ERROR,
    OUTCOME_NOT_STARTED,
    OUTCOME_UNSCORED,
    OUTCOME_WRONG,
    SEMANTIC_UNKNOWN,
    SEMANTIC_VERIFIED,
    BinaryRobustPolicy,
    LabelStore,
    OutcomeStore,
    PanelCoverage,
    PromptOutcome,
    build_label_store,
    build_panel_label,
    check_policy_consistency,
    pair_drop,
    panel_max_drop,
    policy_hash,
)

POLICY = "policy-a"


def outcome(prompt_id: str, correct: int, wrong: int, **extra) -> PromptOutcome:
    """counts 합이 planned_trials와 같은 완전한 기록을 만듭니다."""
    counts = {OUTCOME_CORRECT: correct, OUTCOME_WRONG: wrong}
    counts.update(extra)
    planned = sum(counts.values())
    return PromptOutcome(
        prompt_id=prompt_id,
        counts=counts,
        planned_trials=planned,
        completed_trials=planned - counts.get(OUTCOME_NOT_STARTED, 0),
        policy_hash=POLICY,
        scorer_version="2",
    )


def evidence(slot_id: str, seed: int = 0) -> dict:
    """같은 trajectory 이어받기 증거."""
    return {
        slot_id: {
            "request_id": slot_id,
            "seed": seed,
            "prompt_hash": "ph",
            "policy_hash": POLICY,
            "token_prefix_hash": "tp",
        }
    }


# --------------------------------------------------------------------------------------
# counts와 bounds
# --------------------------------------------------------------------------------------


def test_cap_hit_and_unscored_are_not_merged_into_wrong():
    record = outcome("p", 5, 1, **{OUTCOME_CAP_HIT: 1, OUTCOME_UNSCORED: 1})
    assert record.counts[OUTCOME_WRONG] == 1
    assert record.counts[OUTCOME_CAP_HIT] == 1
    assert record.n_resolved == 6
    assert record.n_unresolved == 2
    assert not record.fully_resolved
    assert record.p_hat() is None  # 미확정이 남으면 point estimate를 만들지 않습니다


def test_p_hat_only_when_every_planned_trial_is_resolved():
    assert outcome("p", 6, 2).p_hat() == pytest.approx(0.75)
    # 미기록 slot은 명시적으로 not_started여야 합니다.
    partial = PromptOutcome(
        prompt_id="p",
        counts={OUTCOME_CORRECT: 3, OUTCOME_WRONG: 1, OUTCOME_NOT_STARTED: 4},
        planned_trials=8,
        completed_trials=4,
        policy_hash=POLICY,
    )
    assert not partial.fully_resolved
    assert partial.p_hat() is None
    assert partial.n_not_started == 4


def test_counts_must_sum_to_planned_trials():
    """기록되지 않은 planned slot은 불완전 입력으로 거부합니다."""
    with pytest.raises(ValueError, match="must equal planned_trials"):
        PromptOutcome(
            prompt_id="p", counts={OUTCOME_CORRECT: 1}, planned_trials=4, completed_trials=1
        )
    with pytest.raises(ValueError, match="non-negative"):
        PromptOutcome(
            prompt_id="p",
            counts={OUTCOME_CORRECT: -1, OUTCOME_NOT_STARTED: 2},
            planned_trials=1,
            completed_trials=0,
        )
    # 명시적 import 규칙으로만 채웁니다.
    filled = PromptOutcome.from_dict(
        {"prompt_id": "p", "counts": {"C": 1}, "planned_trials": 4, "completed_trials": 1},
        fill_missing_as_not_started=True,
    )
    assert filled.n_not_started == 3


def test_unrecorded_slots_do_not_pin_the_upper_bound():
    """planned=4, C=1 뿐이면 upper가 0.25로 굳으면 안 됩니다."""
    record = PromptOutcome(
        prompt_id="p",
        counts={OUTCOME_CORRECT: 1, OUTCOME_NOT_STARTED: 3},
        planned_trials=4,
        completed_trials=1,
        policy_hash=POLICY,
    )
    lower, upper = record.bounds()
    assert lower == pytest.approx(0.25)
    assert upper == pytest.approx(1.0)


def test_zero_planned_trials_have_undefined_probability():
    empty = PromptOutcome(
        prompt_id="z", counts={}, planned_trials=0, completed_trials=0, policy_hash=POLICY
    )
    assert empty.bounds() is None
    assert empty.p_hat() is None


def test_completed_and_resolved_are_different():
    record = outcome("p", 3, 1, **{OUTCOME_CAP_HIT: 2, OUTCOME_UNSCORED: 2})
    assert record.completed_trials == 8  # generation이 끝난 slot 수
    assert record.n_resolved == 4  # 점수가 확정된 slot 수 (C + W)
    assert record.n_unresolved == 4


def test_bounds_are_not_a_confidence_interval():
    record = outcome("p", 4, 2, **{OUTCOME_CAP_HIT: 2})
    lower, upper = record.bounds()
    assert lower == pytest.approx(4 / 8)
    assert upper == pytest.approx(6 / 8)  # unknown이 모두 정답인 최악/최선 경계


def test_pair_drop_signed_value_and_bounds():
    original = outcome("o", 8, 0)
    variant = outcome("o#v", 4, 4)
    label = pair_drop(
        original, variant, original_id="o", variant_id="o#v", panel_id="o#panel",
        semantic_valid=SEMANTIC_VERIFIED,
    )
    assert label.signed_drop == pytest.approx(0.5)
    assert label.resolved and label.has_drop
    assert label.drop_lower == pytest.approx(1.0 - 0.5)
    assert label.drop_upper == pytest.approx(1.0 - 0.5)


def test_negative_signed_drop_is_preserved():
    """variant가 원본보다 좋아진 경우도 그대로 둡니다."""
    label = pair_drop(
        outcome("o", 4, 4), outcome("o#v", 8, 0), original_id="o", variant_id="o#v",
        panel_id="p", semantic_valid=SEMANTIC_VERIFIED,
    )
    assert label.signed_drop == pytest.approx(-0.5)


def test_unresolved_pair_has_no_drop_but_stays_usable_for_flow():
    label = pair_drop(
        outcome("o", 8, 0),
        outcome("o#v", 3, 3, **{OUTCOME_CAP_HIT: 2}),
        original_id="o", variant_id="o#v", panel_id="p", semantic_valid=SEMANTIC_VERIFIED,
    )
    assert label.signed_drop is None
    assert label.exclusion_reason == "unresolved_trials"
    assert label.drop_lower is not None and label.drop_upper is not None
    # Page/semantic validity가 유효하면 flow에는 쓸 수 있습니다.
    assert label.usable_for_flow


def test_unverified_semantics_blocks_behavior_supervision():
    label = pair_drop(
        outcome("o", 8, 0), outcome("o#v", 4, 4), original_id="o", variant_id="o#v",
        panel_id="p", semantic_valid=SEMANTIC_UNKNOWN,
    )
    assert label.signed_drop is None
    assert label.exclusion_reason == "semantic_unknown"
    assert not label.usable_for_flow


def test_pair_drop_requires_the_same_policy():
    other = outcome("o#v", 4, 4)
    other.policy_hash = "policy-b"
    with pytest.raises(ValueError, match="same model/thinking/sampling/scorer policy"):
        pair_drop(
            outcome("o", 8, 0), other, original_id="o", variant_id="o#v", panel_id="p"
        )


# --------------------------------------------------------------------------------------
# panel label
# --------------------------------------------------------------------------------------


def test_partial_panel_has_no_max_drop_target():
    labels = [
        pair_drop(outcome("o", 8, 0), outcome(f"o#v{i}", 4, 4), original_id="o",
                  variant_id=f"o#v{i}", panel_id="p", semantic_valid=SEMANTIC_VERIFIED)
        for i in range(2)
    ]
    partial = PanelCoverage(expected_members=["o#v0", "o#v1", "o#v2"],
                            actual_members=["o#v0", "o#v1"])
    value, panel_only = panel_max_drop(labels, partial)
    assert value is None and panel_only is False  # 관측 최대값을 full-panel target으로 쓰지 않음
    complete = PanelCoverage(expected_members=["o#v0", "o#v1"],
                             actual_members=["o#v0", "o#v1"])
    value, panel_only = panel_max_drop(labels, complete)
    assert value == pytest.approx(0.5) and panel_only is False


def test_panel_only_target_is_marked_independent():
    coverage = PanelCoverage(expected_members=["a"], actual_members=["a"])
    panel = build_panel_label("o", "p", [], coverage, panel_only_max_drop=0.3)
    assert panel.max_drop == pytest.approx(0.3)
    assert panel.max_drop_is_panel_only is True


def test_robust_label_needs_a_frozen_definition():
    coverage = PanelCoverage(expected_members=["a"], actual_members=["a"])
    # 정의가 없으면 robust label은 null로 남습니다.
    panel = build_panel_label("o", "p", [], coverage, provided_robust_label=1)
    assert panel.robust_label is None and not panel.has_robust
    with pytest.raises(ValueError, match="frozen definition"):
        BinaryRobustPolicy(enabled=True).check()
    policy = BinaryRobustPolicy(enabled=True, definition_id="def1", source="src")
    panel = build_panel_label("o", "p", [], coverage, robust_policy=policy,
                              provided_robust_label=0)
    assert panel.robust_label == 0 and panel.has_robust  # label 0은 실제 label입니다


def test_panel_label_is_not_copied_to_pair_labels(datasets):
    """original-panel non-robust label을 모든 variant의 pair label로 복사하지 않습니다."""
    non_robust = [
        g for g in datasets["train"].groups
        if g.panel_label is not None and g.panel_label.robust_label == 0
    ]
    assert non_robust
    group = non_robust[0]
    drops = [
        label.signed_drop for label in group.pair_labels.values() if label.signed_drop is not None
    ]
    # panel이 non-robust여도 pair drop은 각자의 counts에서 나옵니다.
    assert drops
    assert not all(value == drops[0] for value in drops) or len(set(drops)) == 1
    for label in group.pair_labels.values():
        assert label.variant_id in {v.variant_id for v in group.variants}


# --------------------------------------------------------------------------------------
# loss routing
# --------------------------------------------------------------------------------------


def test_missing_label_is_masked_not_zero(datasets):
    batch = collate_panels(datasets["train"].groups[:4])
    missing = ~batch.drop_mask & batch.panel_mask
    assert bool(missing.any())
    assert bool(torch.isnan(batch.drop_target[missing]).all())  # 0이 아니라 NaN
    assert bool(torch.isnan(batch.robust_target[~batch.robust_mask]).all())


def test_label_zero_is_used_as_a_real_label(datasets):
    from aimo.data import OriginalGroup
    from aimo.labels import PanelLabel

    group = datasets["train"].groups[0]
    zero_panel = OriginalGroup(
        original_id=group.original_id,
        original=group.original,
        variants=group.variants,
        panel_id=group.panel_id,
        pair_labels=group.pair_labels,
        panel_label=PanelLabel(
            original_id=group.original_id, panel_id=group.panel_id, robust_label=0,
            robust_source="test", robust_definition="def",
        ),
    )
    batch = collate_panels([zero_panel])
    assert bool(batch.robust_mask[0])
    assert float(batch.robust_target[0]) == 0.0


def test_max_drop_loss_is_off_for_pair_derived_targets(datasets):
    batch = collate_panels(datasets["train"].groups[:6])
    assert bool(batch.max_drop_available.any())
    # pair counts에서 파생된 값은 loss routing mask에서 빠집니다 (이중 감독 방지).
    assert not bool(batch.max_drop_mask.any())


def test_behavior_loss_counts_only_valid_labels(datasets, stats):
    from aimo.config import config_from_dict
    from aimo.losses import behavior_loss
    from aimo.model import build_model
    from helpers import tiny_payload

    train = datasets["train"]
    model = build_model(
        config_from_dict(tiny_payload(model={"name": "joint"})),
        train.hidden_size, train.n_blocks, train.n_landmarks,
    )
    batch = collate_panels(train.groups[:4])
    terms = behavior_loss(model, batch, stats)
    assert terms.n_pairs_with_drop == int(batch.drop_mask.sum())
    assert terms.n_panels_with_robust == int(batch.robust_mask.sum())
    assert terms.n_panels_with_max_drop == 0
    assert not terms.max_drop.defined
    assert terms.pair_drop.defined and terms.robust.defined


# --------------------------------------------------------------------------------------
# OutcomeStore merge와 policy 일관성
# --------------------------------------------------------------------------------------


def test_new_only_merge_does_not_touch_existing_records():
    store = OutcomeStore()
    store.merge([outcome("a", 8, 0)])
    report = store.merge([outcome("a", 0, 8), outcome("b", 4, 4)], mode="new_only")
    assert report["added"] == 1 and report["skipped"] == 1
    assert store.outcomes["a"].n_correct == 8


def test_exact_continuation_needs_trajectory_evidence():
    """counts 단조성만으로 continuation을 인정하지 않습니다."""
    store = OutcomeStore()
    aggregate = outcome("a", 5, 1, **{OUTCOME_CAP_HIT: 2})
    store.merge([aggregate])
    # 증거가 없는 기존 aggregate record에는 이력을 만들어내지 않습니다.
    with pytest.raises(ValueError, match="per-slot trajectory evidence"):
        store.merge([outcome("a", 6, 2)], mode="exact_continuation")

    with_evidence = outcome("b", 5, 1, **{OUTCOME_CAP_HIT: 2})
    with_evidence.slot_evidence = evidence("b#s0")
    store2 = OutcomeStore()
    store2.merge([with_evidence])
    continued = outcome("b", 6, 2)
    continued.slot_evidence = evidence("b#s0")
    store2.merge([continued], mode="exact_continuation")
    assert store2.outcomes["b"].fully_resolved
    # seed가 달라지면 같은 trajectory가 아닙니다.
    other = outcome("b", 7, 1)
    other.slot_evidence = evidence("b#s0", seed=99)
    with pytest.raises(ValueError, match="independent re-run"):
        store2.merge([other], mode="exact_continuation")


def test_fill_not_started_keeps_completed_slots():
    """미시작 slot만 채우고 기존 X를 새 성공으로 대체하지 않습니다."""
    store = OutcomeStore()
    store.merge([outcome("a", 1, 0, **{OUTCOME_CAP_HIT: 1, OUTCOME_NOT_STARTED: 2})])
    store.merge([outcome("a", 2, 0)], mode="fill_not_started")
    record = store.outcomes["a"]
    assert record.counts[OUTCOME_CORRECT] == 3
    assert record.counts[OUTCOME_CAP_HIT] == 1  # 기존 cap-hit 유지
    assert record.n_not_started == 0
    assert record.planned_trials == 4
    # 완료된 slot을 다시 채우려 하면 거부합니다.
    with pytest.raises(ValueError, match="were not_started"):
        store.merge([outcome("a", 1, 0)], mode="fill_not_started")


def test_independent_rerun_goes_to_a_separate_cohort():
    """독립 재실행은 primary 기록을 바꾸지 않고 별도 cohort로 보존합니다."""
    store = OutcomeStore()
    store.merge([outcome("a", 1, 0, **{OUTCOME_CAP_HIT: 3})])
    report = store.merge([outcome("a", 4, 0)], mode="separate_cohort", cohort="rerun1")
    assert report["cohort_records"] == 1
    assert store.outcomes["a"].counts[OUTCOME_CORRECT] == 1  # primary 불변
    assert store.outcomes["a"].counts[OUTCOME_CAP_HIT] == 3
    assert store.cohorts["rerun1"]["a"].counts[OUTCOME_CORRECT] == 4
    with pytest.raises(ValueError, match="needs an explicit cohort name"):
        store.merge([outcome("b", 1, 0)], mode="separate_cohort")


def test_outcome_store_rejects_a_different_policy():
    store = OutcomeStore()
    store.merge([outcome("a", 8, 0)])
    other = outcome("b", 8, 0)
    other.policy_hash = "policy-b"
    with pytest.raises(ValueError, match="keep one policy per store"):
        store.merge([other])


def test_outcome_store_rejects_a_different_scorer_version():
    """이전 결과를 새 scorer 결과로 덮어쓰지 않습니다."""
    store = OutcomeStore()
    store.merge([outcome("a", 8, 0)])
    other = outcome("b", 8, 0)
    other.scorer_version = "1"
    with pytest.raises(ValueError, match="do not overwrite earlier results"):
        store.merge([other])


def test_duplicate_prompt_in_one_batch_is_rejected():
    store = OutcomeStore()
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        store.merge([outcome("a", 8, 0), outcome("a", 4, 4)])


def test_label_store_policy_consistency():
    store = LabelStore(policy_hash=POLICY)
    good = pair_drop(outcome("o", 8, 0), outcome("o#v", 4, 4), original_id="o",
                     variant_id="o#v", panel_id="p", semantic_valid=SEMANTIC_VERIFIED)
    store.pairs["o#v"] = good
    check_policy_consistency(store)
    bad = pair_drop(outcome("o2", 8, 0), outcome("o2#v", 4, 4), original_id="o2",
                    variant_id="o2#v", panel_id="p2", semantic_valid=SEMANTIC_VERIFIED)
    bad.policy_hash = "policy-b"
    store.pairs["o2#v"] = bad
    with pytest.raises(ValueError, match="different model/thinking/sampling policies"):
        check_policy_consistency(store)


def test_build_label_store_keeps_failures_and_reports_coverage():
    from aimo.adapters.perturbation import PairCandidate

    outcomes = OutcomeStore(policy_hash=POLICY)
    outcomes.merge(
        [
            outcome("o", 8, 0),
            outcome("o#v0", 0, 8),  # 완전 실패도 보존합니다
            outcome("o#v1", 8, 0),  # 성능 유지
            outcome("o#v2", 4, 2, **{OUTCOME_INFRA_ERROR: 2}),  # 미확정
        ]
    )
    pairs = [
        PairCandidate(
            pair_id=f"o#v{i}", original_id="o", variant_id=f"o#v{i}", panel_id="o#panel",
            original_text="a", variant_text="b", original_answer="1", variant_answer="1",
            semantic_valid=SEMANTIC_VERIFIED,
        )
        for i in range(3)
    ]
    store = build_label_store(outcomes, pairs)
    report = store.coverage_report()
    assert report["n_pairs"] == 3
    assert report["n_pairs_with_drop"] == 2
    assert report["pair_exclusion_reasons"] == {"unresolved_trials": 1}
    assert store.pairs["o#v0"].signed_drop == pytest.approx(1.0)  # 완전 실패
    assert store.pairs["o#v1"].signed_drop == pytest.approx(0.0)  # 성능 유지
    # robust label은 정의가 없으므로 null입니다.
    assert store.panels["o"].robust_label is None
    # coverage가 불완전하므로 max_drop도 없습니다.
    assert store.panels["o"].max_drop is None


def test_policy_hash_is_stable():
    payload = {"model": "Qwen/Qwen3-4B", "thinking": True}
    assert policy_hash(payload) == policy_hash(dict(reversed(list(payload.items()))))
    assert policy_hash(payload) != policy_hash({**payload, "thinking": False})
