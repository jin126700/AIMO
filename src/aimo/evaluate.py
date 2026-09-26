"""평가 metrics.

집계 단위는 항상 original group입니다. layer나 variant를 독립 원문처럼 세지
않습니다. 불확실성은 original-group bootstrap으로 보고하고, training seed 사이의
분산과는 따로 기록합니다 (bootstrap CI는 seed variance가 아닙니다).

Zero/noise norm에서의 cosine과 relative magnitude는 값을 만들지 않고 undefined로
세어 둡니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .data import (
    NormStats,
    OriginalGroup,
    PageDataset,
    PairBatch,
    collate,
    collate_panels,
    enumerate_pairs,
    group_by_cut,
    swap_panel_support,
    swap_support,
)
from .losses import HUBER_DELTA, target_cell_mask
from .rollout import rollout


@dataclass
class MetricAccumulator:
    """original group별로 값을 모은 뒤 group 단위로 평균/bootstrap합니다."""

    per_group: dict[str, list[float]] = field(default_factory=dict)
    n_undefined: int = 0

    def add(self, group_id: str, value: float) -> None:
        if value != value or math.isinf(value):  # NaN / inf는 undefined로 셉니다.
            self.n_undefined += 1
            return
        self.per_group.setdefault(group_id, []).append(value)

    def skip(self, n: int = 1) -> None:
        self.n_undefined += n

    def group_means(self) -> list[float]:
        return [sum(v) / len(v) for v in self.per_group.values()]

    def summary(self, bootstrap_samples: int = 200, seed: int = 0) -> dict:
        means = self.group_means()
        if not means:
            return {
                "mean": None,
                "n_originals": 0,
                "n_undefined": self.n_undefined,
                "ci95_low": None,
                "ci95_high": None,
            }
        values = torch.tensor(means, dtype=torch.float64)
        out = {
            "mean": float(values.mean()),
            "n_originals": len(means),
            "n_undefined": self.n_undefined,
            "ci95_low": None,
            "ci95_high": None,
        }
        if bootstrap_samples > 0 and len(means) > 1:
            generator = torch.Generator().manual_seed(seed)
            idx = torch.randint(
                0, len(means), (bootstrap_samples, len(means)), generator=generator
            )
            draws = values[idx].mean(dim=1)
            out["ci95_low"] = float(torch.quantile(draws, 0.025))
            out["ci95_high"] = float(torch.quantile(draws, 0.975))
            out["bootstrap"] = "original-group resampling (not training-seed variance)"
        return out


def _masked_mse_per_item(pred: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    m = mask.expand_as(pred).to(pred.dtype)
    dims = tuple(range(1, pred.ndim))
    total = ((pred - target).pow(2) * m).sum(dim=dims)
    count = m.sum(dim=dims)
    return total / count.clamp_min(1.0), count


def _cosine_and_ratio(
    pred: Tensor, target: Tensor, mask: Tensor, floor: float
) -> tuple[list[float], list[float], int]:
    """(landmark, stream)별로 H 방향의 cosine과 크기비를 구합니다.

    어느 한쪽 norm이 floor 이하이면 undefined로 세고 값을 만들지 않습니다.
    """
    cell = mask.squeeze(-1)  # [B, P, 2]
    pred_norm = pred.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    dot = (pred * target).sum(dim=-1)
    usable = cell & (pred_norm > floor) & (target_norm > floor)
    undefined = int((cell & ~usable).sum())
    cos = (dot / (pred_norm * target_norm).clamp_min(floor))[usable]
    ratio = (pred_norm / target_norm.clamp_min(floor))[usable]
    return cos.tolist(), ratio.tolist(), undefined


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    dataset: PageDataset,
    stats: NormStats,
    horizons: tuple[int, ...] = (2, 4),
    bootstrap_samples: int = 200,
    support_swap: bool = True,
    seed: int = 0,
) -> dict:
    """한 split에 대한 결정적 평가. 모든 cut과 sibling 조합을 나열해 씁니다."""
    model.eval()
    n_blocks = dataset.n_blocks
    cuts = list(range(n_blocks))
    samples = enumerate_pairs(dataset, cuts)

    acc = {
        "next_mse_raw": MetricAccumulator(),
        "next_mse_norm": MetricAccumulator(),
        "direction_cosine": MetricAccumulator(),
        "relative_magnitude": MetricAccumulator(),
        "sibling_difference_gain": MetricAccumulator(),
        "identity_next_mse_norm": MetricAccumulator(),
    }
    for horizon in horizons:
        acc[f"rollout_h{horizon}_mse_norm"] = MetricAccumulator()
    if support_swap:
        acc["swapped_support_next_mse_norm"] = MetricAccumulator()
        acc["support_swap_degradation"] = MetricAccumulator()

    for bucket in group_by_cut(samples):
        batch = collate(bucket)
        _accumulate_batch(model, batch, stats, acc, horizons, support_swap)

    results = {
        "split": dataset.split,
        "n_originals": len(dataset.groups),
        "n_pair_cut_samples": len(samples),
        "metrics": {
            name: a.summary(bootstrap_samples=bootstrap_samples, seed=seed)
            for name, a in acc.items()
        },
        "notes": [
            "집계 단위는 original group입니다. layer/variant는 독립 원문이 아닙니다.",
            "bootstrap CI는 original-group resampling이며 training seed variance가 아닙니다.",
            "zero/noise norm의 cosine과 relative magnitude는 undefined로 셉니다.",
        ],
    }
    return results


def _accumulate_batch(
    model: nn.Module,
    batch: PairBatch,
    stats: NormStats,
    acc: dict[str, MetricAccumulator],
    horizons: tuple[int, ...],
    support_swap: bool,
) -> None:
    cut = batch.cut
    mask = target_cell_mask(batch, stats)  # [B, P, 2, 1]
    pred_a_norm = model(batch.input_a, stats)
    pred_b_norm = model(batch.input_b, stats)
    pred_a_raw = stats.denorm_target(pred_a_norm, cut)
    tgt_a_raw = batch.target_a
    tgt_a_norm = stats.norm_target(tgt_a_raw, cut)

    mse_raw, count = _masked_mse_per_item(pred_a_raw, tgt_a_raw, mask)
    mse_norm, _ = _masked_mse_per_item(pred_a_norm, tgt_a_norm, mask)
    floor = stats.floor

    for i, group_id in enumerate(batch.original_ids):
        if count[i] <= 0:
            acc["next_mse_raw"].skip()
            acc["next_mse_norm"].skip()
            continue
        acc["next_mse_raw"].add(group_id, float(mse_raw[i]))
        acc["next_mse_norm"].add(group_id, float(mse_norm[i]))
        if bool(batch.is_identity_a[i]):
            acc["identity_next_mse_norm"].add(group_id, float(mse_norm[i]))

    # 방향과 크기비: (landmark, stream) 단위.
    for i, group_id in enumerate(batch.original_ids):
        cos, ratio, undefined = _cosine_and_ratio(
            pred_a_raw[i : i + 1], tgt_a_raw[i : i + 1], mask[i : i + 1], floor
        )
        acc["direction_cosine"].skip(undefined)
        acc["relative_magnitude"].skip(undefined)
        for value in cos:
            acc["direction_cosine"].add(group_id, value)
        for value in ratio:
            acc["relative_magnitude"].add(group_id, value)

    # sibling-difference gain: zero 차이 예측 대비 상대적 감소량.
    pred_diff = pred_a_norm - pred_b_norm
    true_diff = stats.norm_sibling(batch.target_a - batch.target_b, cut)
    diff_mse, _ = _masked_mse_per_item(pred_diff, true_diff, mask)
    zero_mse, _ = _masked_mse_per_item(torch.zeros_like(true_diff), true_diff, mask)
    for i, group_id in enumerate(batch.original_ids):
        if not bool(batch.has_sibling[i]) or float(zero_mse[i]) <= floor:
            acc["sibling_difference_gain"].skip()
            continue
        acc["sibling_difference_gain"].add(
            group_id, 1.0 - float(diff_mse[i]) / float(zero_mse[i])
        )

    # rollout: horizon별 D_hat 오차.
    n_blocks = batch.input_a.n_blocks
    result = rollout(model, batch.input_a, stats, max(horizons))
    for horizon in horizons:
        depth = cut + horizon
        name = f"rollout_h{horizon}_mse_norm"
        pred = result.state_diff_at(depth) if depth <= n_blocks else None
        if pred is None or not bool(stats.rollout_active[depth] if depth <= n_blocks else False):
            acc[name].skip(len(batch.original_ids))
            continue
        pred_n = stats.norm_rollout(pred, depth)
        true_n = stats.norm_rollout(batch.future_state_diff_a[:, depth], depth)
        roll_mask = batch.valid.view(batch.valid.shape[0], -1, 1)
        per_item, roll_count = _masked_mse_per_item(pred_n, true_n, roll_mask)
        for i, group_id in enumerate(batch.original_ids):
            if roll_count[i] <= 0:
                acc[name].skip()
                continue
            acc[name].add(group_id, float(per_item[i]))

    # support-swap: 같은 original의 sibling 사이에서만 support를 바꿉니다.
    if support_swap:
        swapped = swap_support(batch, torch.Generator().manual_seed(0))
        pred_swapped = model(swapped.input_a, stats)
        swap_mse, swap_count = _masked_mse_per_item(pred_swapped, tgt_a_norm, mask)
        for i, group_id in enumerate(batch.original_ids):
            if not bool(batch.has_sibling[i]) or swap_count[i] <= 0:
                acc["swapped_support_next_mse_norm"].skip()
                acc["support_swap_degradation"].skip()
                continue
            acc["swapped_support_next_mse_norm"].add(group_id, float(swap_mse[i]))
            acc["support_swap_degradation"].add(
                group_id, float(swap_mse[i]) - float(mse_norm[i])
            )


# --------------------------------------------------------------------------------------
# Behavior 평가
# --------------------------------------------------------------------------------------


def _auroc(scores: list[float], labels: list[int]) -> float | None:
    """rank 기반 AUROC. 한 class만 있으면 undefined입니다."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return None
    values = torch.tensor(scores, dtype=torch.float64)
    order = values.argsort()
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(1, len(values) + 1, dtype=torch.float64)
    # 동점은 평균 rank로 처리합니다.
    unique = values.unique()
    for value in unique.tolist():
        mask = values == value
        if int(mask.sum()) > 1:
            ranks[mask] = ranks[mask].mean()
    label_t = torch.tensor(labels, dtype=torch.float64)
    sum_pos_ranks = float(ranks[label_t == 1].sum())
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _spearman(a: list[float], b: list[float]) -> float | None:
    """작은 panel 안의 순위 상관. 값이 2개 미만이거나 한쪽이 상수면 undefined입니다."""
    if len(a) < 2:
        return None
    ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    if float(ta.std()) == 0.0 or float(tb.std()) == 0.0:
        return None

    def rank(t: torch.Tensor) -> torch.Tensor:
        order = t.argsort()
        out = torch.empty_like(t)
        out[order] = torch.arange(1, len(t) + 1, dtype=torch.float64)
        return out

    ra, rb = rank(ta), rank(tb)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float((ra.pow(2).sum() * rb.pow(2).sum()).sqrt())
    if denom == 0.0:
        return None
    return float((ra * rb).sum()) / denom


def _classification_metrics(probs: list[float], labels: list[int]) -> dict:
    """robust classification 지표. 양 class recall/precision과 confusion counts를 함께 냅니다."""
    if not probs:
        return {"n_panels": 0, "note": "no labeled panel"}
    pred = [1 if p >= 0.5 else 0 for p in probs]
    tp = sum(1 for p, y in zip(pred, labels, strict=True) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(pred, labels, strict=True) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(pred, labels, strict=True) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(pred, labels, strict=True) if p == 0 and y == 1)
    n = len(labels)
    n_pos, n_neg = tp + fn, tn + fp
    recall_pos = tp / n_pos if n_pos else None
    recall_neg = tn / n_neg if n_neg else None
    eps = 1e-12
    log_loss = -sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        for p, y in zip(probs, labels, strict=True)
    ) / n
    balanced = (
        (recall_pos + recall_neg) / 2 if recall_pos is not None and recall_neg is not None else None
    )
    return {
        "n_panels": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": balanced,
        "recall_robust": recall_pos,
        "recall_non_robust": recall_neg,
        "precision_robust": tp / (tp + fp) if (tp + fp) else None,
        "precision_non_robust": tn / (tn + fn) if (tn + fn) else None,
        "auroc": _auroc(probs, labels),
        "brier": sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / n,
        "log_loss": log_loss,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "predicted_all_one_class": len(set(pred)) == 1,
        "majority_class_accuracy": max(n_pos, n_neg) / n,
    }


@torch.no_grad()
def evaluate_behavior(
    model: nn.Module,
    dataset: PageDataset,
    stats: NormStats,
    bootstrap_samples: int = 200,
    seed: int = 0,
    support_swap: bool = True,
    microbatch: int = 4,
) -> dict:
    """behavior view 평가. 집계 단위는 original panel입니다."""
    model.eval()
    acc = {
        "pair_drop_mae": MetricAccumulator(),
        "pair_drop_huber": MetricAccumulator(),
        "within_panel_pred_spread": MetricAccumulator(),
        "within_panel_rank_corr": MetricAccumulator(),
        "drop_bound_width": MetricAccumulator(),
        "max_drop_mae_diagnostic": MetricAccumulator(),
    }
    if support_swap:
        acc["pair_support_swap_drop_shift"] = MetricAccumulator()
        acc["pair_support_swap_mae_increase"] = MetricAccumulator()
    probs: list[float] = []
    labels: list[int] = []
    panel_ids: list[str] = []
    permutation_shift = 0.0
    n_supervised_pairs = 0
    n_pairs = 0

    chunks = [
        dataset.groups[i : i + microbatch] for i in range(0, len(dataset.groups), microbatch)
    ]
    for chunk in chunks:
        batch = collate_panels(chunk)
        out = model.forward_behavior(batch.inputs, stats)
        panel = model.panel_outputs(out, batch.pair_panel, batch.pair_slot, batch.panel_mask)
        pred = panel.pair_drop  # [B, M], padding은 NaN

        for b, group in enumerate(chunk):
            group_id = group.original_id
            slots = torch.nonzero(batch.drop_mask[b]).flatten().tolist()
            n_pairs += int(batch.panel_mask[b].sum())
            n_supervised_pairs += len(slots)
            true_vals, pred_vals = [], []
            for slot in slots:
                target = float(batch.drop_target[b, slot])
                predicted = float(pred[b, slot])
                acc["pair_drop_mae"].add(group_id, abs(predicted - target))
                diff = abs(predicted - target)
                huber = (
                    0.5 * diff**2 / HUBER_DELTA if diff < HUBER_DELTA else diff - 0.5 * HUBER_DELTA
                )
                acc["pair_drop_huber"].add(group_id, huber)
                true_vals.append(target)
                pred_vals.append(predicted)
            if len(slots) >= 2:
                spread = float(torch.tensor(pred_vals, dtype=torch.float64).std())
                acc["within_panel_pred_spread"].add(group_id, spread)
                corr = _spearman(pred_vals, true_vals)
                if corr is None:
                    acc["within_panel_rank_corr"].skip()
                else:
                    acc["within_panel_rank_corr"].add(group_id, corr)
            # sampling uncertainty: 저장된 bound 폭
            for variant in group.variants:
                label = group.pair_labels.get(variant.variant_id)
                if label is None or label.drop_lower is None or label.drop_upper is None:
                    continue
                acc["drop_bound_width"].add(group_id, label.drop_upper - label.drop_lower)
            if bool(batch.robust_mask[b]) and bool(panel.panel_valid[b]):
                probs.append(float(panel.robust_prob[b]))
                labels.append(int(batch.robust_target[b]))
                panel_ids.append(group_id)
            if bool(batch.max_drop_available[b]) and bool(panel.panel_valid[b]):
                acc["max_drop_mae_diagnostic"].add(
                    group_id, abs(float(panel.max_drop[b]) - float(batch.max_drop_target[b]))
                )

        # panel permutation invariance: 같은 panel의 variants 순서를 뒤집어 확인합니다.
        flipped = [
            OriginalGroup(
                original_id=group.original_id,
                original=group.original,
                variants=list(reversed(group.variants)),
                panel_id=group.panel_id,
                pair_labels=group.pair_labels,
                panel_label=group.panel_label,
                metadata=group.metadata,
            )
            for group in chunk
        ]
        flipped_batch = collate_panels(flipped)
        flipped_out = model.forward_behavior(flipped_batch.inputs, stats)
        flipped_panel = model.panel_outputs(
            flipped_out,
            flipped_batch.pair_panel,
            flipped_batch.pair_slot,
            flipped_batch.panel_mask,
        )
        both = panel.panel_valid & flipped_panel.panel_valid
        if bool(both.any()):
            shift = (
                (panel.robust_prob[both] - flipped_panel.robust_prob[both]).abs().max()
            )
            permutation_shift = max(permutation_shift, float(shift))

        # pair support-swap: target variant를 고정하고 sibling의 Page를 대입합니다.
        if support_swap:
            swapped = swap_panel_support(batch, target_slot=0)
            swapped_out = model.forward_behavior(swapped.inputs, stats)
            swapped_panel = model.panel_outputs(
                swapped_out, swapped.pair_panel, swapped.pair_slot, swapped.panel_mask
            )
            for b, group in enumerate(chunk):
                if int(batch.panel_mask[b].sum()) < 2 or not bool(batch.drop_mask[b, 0]):
                    acc["pair_support_swap_drop_shift"].skip()
                    acc["pair_support_swap_mae_increase"].skip()
                    continue
                target = float(batch.drop_target[b, 0])
                correct = float(pred[b, 0])
                swapped_value = float(swapped_panel.pair_drop[b, 0])
                acc["pair_support_swap_drop_shift"].add(
                    group.original_id, abs(swapped_value - correct)
                )
                acc["pair_support_swap_mae_increase"].add(
                    group.original_id,
                    abs(swapped_value - target) - abs(correct - target),
                )

    trained = model.trained_heads() if hasattr(model, "trained_heads") else {}
    results = {
        "split": dataset.split,
        "n_originals": len(dataset.groups),
        "n_pairs": n_pairs,
        "n_supervised_pairs": n_supervised_pairs,
        "supervised_pair_coverage": (n_supervised_pairs / n_pairs) if n_pairs else 0.0,
        "trained_heads": trained,
        "metrics": {
            name: a.summary(bootstrap_samples=bootstrap_samples, seed=seed)
            for name, a in acc.items()
        },
        "robust_classification": _classification_metrics(probs, labels),
        "panel_permutation_max_prob_shift": permutation_shift,
        "notes": [
            "집계 단위는 original panel입니다. layer/variant를 독립 문제로 세지 않습니다.",
            "panel 재정렬 불변성은 set pooling 검사이며 pair 정보 사용 여부의 ablation이 아닙니다.",
            "pair support-swap은 target variant를 고정한 채 sibling Page를 대입한 결과입니다.",
            "학습되지 않은 head의 출력은 검증된 robust probability가 아닙니다.",
        ],
    }
    if trained and not trained.get("robust", False):
        results["robust_classification"]["untrained_head"] = True
        results["robust_classification"]["warning"] = (
            "robust head was never trained on a real label; these numbers are from an "
            "untrained head"
        )
    if labels and len(set(labels)) == 1:
        results["robust_classification"]["single_class_only"] = True
    return results
