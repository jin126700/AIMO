"""Microbatch 집계·metric·device·budget regression test."""

from __future__ import annotations

import pytest
import torch

from aimo.config import config_from_dict
from aimo.data import collate, collate_panels, group_by_cut, sample_pairs
from aimo.losses import (
    HUBER_DELTA,
    behavior_loss,
    behavior_term_counts,
    compute_loss,
    flow_term_counts,
    huber_elementwise,
)
from aimo.model import build_model
from helpers import tiny_payload


def joint_model(datasets, dropout: float = 0.0):
    cfg = config_from_dict(
        tiny_payload(model={"name": "joint", "dropout": dropout}, train={"task": "joint"})
    )
    train = datasets["train"]
    return build_model(cfg, train.hidden_size, train.n_blocks, train.n_landmarks)


def _behavior_objective(model, groups, stats, micro: int) -> float:
    chunks = [groups[i : i + micro] for i in range(0, len(groups), micro)]
    batches = [collate_panels(chunk) for chunk in chunks]
    denominators: dict[str, int] = {}
    for batch in batches:
        for key, value in behavior_term_counts(batch).items():
            denominators[key] = denominators.get(key, 0) + value
    total = 0.0
    for batch in batches:
        total += float(behavior_loss(model, batch, stats, denominators=denominators).total)
    return total


def test_behavior_objective_is_microbatch_invariant(datasets, stats):
    """missing label이 있어도 microbatch 크기가 effective objective를 바꾸지 않습니다."""
    model = joint_model(datasets)
    model.eval()
    groups = datasets["train"].groups
    with torch.no_grad():
        values = [_behavior_objective(model, groups, stats, micro) for micro in (1, 2, 3, 6)]
    assert max(values) - min(values) < 1e-6


def test_gradients_are_microbatch_invariant(datasets, stats):
    """같은 effective batch에서 microbatch=1/2/4의 gradient가 허용오차 안에서 같습니다."""
    groups = datasets["train"].groups

    def grads(micro: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(0)
        model = joint_model(datasets)
        model.train()  # dropout=0.0
        chunks = [groups[i : i + micro] for i in range(0, len(groups), micro)]
        behavior_batches = [collate_panels(chunk) for chunk in chunks]
        b_denoms: dict[str, int] = {}
        for batch in behavior_batches:
            for key, value in behavior_term_counts(batch).items():
                b_denoms[key] = b_denoms.get(key, 0) + value
        generator = torch.Generator().manual_seed(11)
        samples = sample_pairs(datasets["train"], generator, 1)
        flow_batches = [
            collate(bucket[i : i + micro])
            for bucket in group_by_cut(samples)
            for i in range(0, len(bucket), micro)
        ]
        f_denoms: dict[str, int] = {}
        for batch in flow_batches:
            for key, value in flow_term_counts(batch, stats, (2, 4)).items():
                f_denoms[key] = f_denoms.get(key, 0) + value
        for batch in behavior_batches:
            behavior_loss(model, batch, stats, denominators=b_denoms).total.backward()
        for batch in flow_batches:
            (0.1 * compute_loss(model, batch, stats, denominators=f_denoms).total).backward()
        return {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

    reference = grads(1)
    scale = max(float(value.abs().max()) for value in reference.values())
    for micro in (2, 4):
        current = grads(micro)
        worst = max(float((reference[key] - current[key]).abs().max()) for key in reference)
        assert worst / scale < 1e-5


def test_terms_without_labels_are_excluded_without_nan_gradients(datasets, stats):
    """label이 전혀 없는 항은 NaN gradient 없이 제외됩니다."""
    model = joint_model(datasets)
    model.train()
    batch = collate_panels(datasets["train"].groups[:4])
    terms = behavior_loss(model, batch, stats, use_max_drop=True)
    assert not terms.max_drop.defined  # panel-only max-drop target이 없습니다
    terms.total.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_validation_uses_the_same_sum_count_aggregation(datasets, stats):
    from aimo.train import validation_metrics

    model = joint_model(datasets)
    values = []
    for micro in (1, 2, 3):
        cfg = config_from_dict(
            tiny_payload(
                model={"name": "joint", "dropout": 0.0},
                train={"task": "joint", "microbatch_originals": micro},
            )
        )
        values.append(validation_metrics(model, datasets["validation"], stats, cfg))
    for key in ("behavior_total", "L_pair_drop", "L_robust", "flow_total", "L_next"):
        seen = [value[key] for value in values]
        assert max(seen) - min(seen) < 1e-6, key


def test_huber_helper_is_shared_between_train_and_eval():
    """evaluation이 training과 다른 SmoothL1 스케일을 쓰지 않습니다."""
    diff = torch.tensor(0.5)
    zero = torch.tensor(0.0)
    shared = float(huber_elementwise(diff, zero))
    expected = HUBER_DELTA * (0.5 - 0.5 * HUBER_DELTA)
    assert shared == pytest.approx(expected)
    small = float(huber_elementwise(torch.tensor(0.05), zero))
    assert small == pytest.approx(0.5 * 0.05**2)


def test_spearman_handles_ties_and_constants():
    from aimo.evaluate import _average_rank, _spearman

    assert _average_rank(torch.tensor([1.0, 2.0, 2.0, 3.0])).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert _spearman([1, 2, 2, 3], [1, 2, 3, 4]) is not None
    assert _spearman([1, 1, 1], [1, 2, 3]) is None  # 한쪽이 상수면 undefined
    assert _spearman([1], [2]) is None


def test_support_swap_covers_every_valid_target_variant(datasets, stats):
    """slot 0만 보지 않고 유효한 target variant 전부를 평가합니다."""
    from aimo.evaluate import evaluate_behavior

    model = joint_model(datasets)
    results = evaluate_behavior(model, datasets["known_test"], stats, bootstrap_samples=0)
    swap = results["metrics"]["pair_support_swap_mae_increase"]
    # 모든 원문이 기여해야 합니다 (slot0에 label이 없는 원문도 다른 slot으로 기여).
    n_panels_with_multiple = sum(
        1
        for group in datasets["known_test"].groups
        if sum(1 for v in group.variants if not v.is_identity) >= 2
        and any(label.has_drop for label in group.pair_labels.values())
    )
    assert swap["n_originals"] == n_panels_with_multiple


def test_untrained_robust_head_is_not_reported_as_canonical(datasets, stats):
    from aimo.evaluate import evaluate_behavior

    model = joint_model(datasets)
    results = evaluate_behavior(model, datasets["known_test"], stats, bootstrap_samples=0)
    assert results["robust_classification"] is None
    assert results["robust_head_status"] == "untrained"
    # 원시 score는 debug 출력으로만 남습니다.
    assert "accuracy" in results["debug_untrained_robust_classification"]
    assert results["max_drop_source"] == "derived_from_pair_predictions"
    model.mark_trained("robust")
    trained = evaluate_behavior(model, datasets["known_test"], stats, bootstrap_samples=0)
    assert trained["robust_classification"] is not None
    assert trained["robust_head_status"] == "trained"


# --------------------------------------------------------------------------------------
# device 경로
# --------------------------------------------------------------------------------------


def test_cuda_request_without_cuda_is_an_explicit_error():
    from aimo.runtime import resolve_device

    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda:0")
    with pytest.raises(ValueError, match="unsupported device"):
        resolve_device("tpu")


def test_microbatch_moves_without_touching_the_whole_dataset(datasets, stats):
    """batch와 NormStats가 device 이동을 지원하고 원본은 그대로입니다."""
    device = torch.device("cpu")
    batch = collate_panels(datasets["train"].groups[:2])
    moved = batch.to(device)
    assert moved is batch  # 같은 device면 복제하지 않습니다
    assert stats.to(device) is stats
    generator = torch.Generator().manual_seed(0)
    flow = collate(group_by_cut(sample_pairs(datasets["train"], generator, 1))[0])
    assert flow.to(device) is flow
    # 전체 dataset이 아니라 microbatch만 다룹니다.
    assert moved.inputs.batch_size <= sum(
        1 for g in datasets["train"].groups[:2] for v in g.variants
    )


def test_zero_weight_objective_is_not_marked_trained(tmp_path, datasets):
    """가중치가 0인 항은 objective에 기여하지 않으므로 학습된 head로 표시하지 않습니다."""
    from aimo.train import train

    payload = tiny_payload(
        model={"name": "joint"},
        train={
            "task": "joint",
            "w_robust": 0.0,
            "select_metric": "L_pair_drop",
            "max_epochs": 1,
            "patience": 1,
        },
    )
    payload["paths"] = {"output_root": str(tmp_path)}
    payload["run"] = {"run_id": "zeroweight", "seed": 0}
    cfg = config_from_dict(payload)
    summary = train(cfg, datasets)
    assert summary["trained_heads"]["pair_drop"] is True
    assert summary["trained_heads"]["robust"] is False
    assert summary["supervision"]["enabled_behavior_objectives"] == ["pair_drop"]
