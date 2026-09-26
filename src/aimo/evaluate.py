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
    PageDataset,
    PairBatch,
    collate,
    enumerate_pairs,
    group_by_cut,
    swap_support,
)
from .losses import target_cell_mask
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
