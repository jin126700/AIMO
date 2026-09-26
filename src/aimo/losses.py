"""Loss 항과 masked normalized MSE.

    L_next   = normalized MSE(V_hat, V)
    L_within = normalized MSE(V_hat_a - V_hat_b, V_a - V_b)
    L_roll   = normalized MSE(D_hat_future, D_future), horizon 2/4 평균
    L        = w_next * L_next + w_within * L_within + w_roll * L_roll

기본 가중치는 1 / 1 / 0.25입니다. 모든 항은 원문(original) 단위로 먼저 평균한 뒤
원문끼리 동일 가중치로 평균합니다. valid landmark와 active scale만 세고, inactive
depth/stream은 분모에서 제외합니다.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .data import NormStats, PairBatch
from .rollout import rollout


@dataclass
class TermValue:
    """항 하나의 값과 기여한 원문 수. count=0이면 항이 정의되지 않습니다."""

    value: Tensor
    count: int

    @property
    def defined(self) -> bool:
        return self.count > 0

    def item(self) -> float:
        return float(self.value) if self.count > 0 else float("nan")


def _per_item_mean(
    pred: Tensor, target: Tensor, cell_mask: Tensor
) -> tuple[Tensor, Tensor]:
    """원문별 masked MSE와 원문별 유효 cell 수를 돌려줍니다.

    pred/target: [B, ...], cell_mask: pred에 broadcast 가능한 bool.
    """
    mask = cell_mask.expand_as(pred).to(pred.dtype)
    sq = (pred - target).pow(2) * mask
    dims = tuple(range(1, pred.ndim))
    total = sq.sum(dim=dims)
    count = mask.sum(dim=dims)
    per_item = total / count.clamp_min(1.0)
    return per_item, count


def _reduce_items(per_item: Tensor, count: Tensor, item_mask: Tensor | None = None) -> TermValue:
    keep = count > 0
    if item_mask is not None:
        keep = keep & item_mask
    n = int(keep.sum())
    if n == 0:
        return TermValue(value=per_item.sum() * 0.0, count=0)
    return TermValue(value=(per_item * keep.to(per_item.dtype)).sum() / n, count=n)


def target_cell_mask(batch: PairBatch, stats: NormStats) -> Tensor:
    """[B, P, 2, 1] bool: valid landmark AND active target scale."""
    valid = batch.valid.view(batch.valid.shape[0], -1, 1, 1)
    active = stats.target_active[batch.cut].view(1, 1, 2, 1)
    return valid & active


def next_update_loss(
    pred_a_norm: Tensor,
    pred_b_norm: Tensor,
    batch: PairBatch,
    stats: NormStats,
) -> TermValue:
    """L_next. a와 b 예측을 같은 가중치로 함께 평균합니다."""
    mask = target_cell_mask(batch, stats)
    tgt_a = stats.norm_target(batch.target_a, batch.cut)
    tgt_b = stats.norm_target(batch.target_b, batch.cut)
    mean_a, count_a = _per_item_mean(pred_a_norm, tgt_a, mask)
    term_a = _reduce_items(mean_a, count_a)
    mean_b, count_b = _per_item_mean(pred_b_norm, tgt_b, mask)
    term_b = _reduce_items(mean_b, count_b, item_mask=batch.has_sibling)
    if not term_b.defined:
        return term_a
    if not term_a.defined:
        return term_b
    return TermValue(value=0.5 * (term_a.value + term_b.value), count=term_a.count)


def within_original_loss(
    pred_a_norm: Tensor,
    pred_b_norm: Tensor,
    batch: PairBatch,
    stats: NormStats,
) -> TermValue:
    """L_within. 같은 original의 서로 다른 stable variants 사이 차이를 맞춥니다.

    같은 cut과 common valid cells만 사용합니다. sibling이 없는 원문은 제외합니다.
    """
    mask = target_cell_mask(batch, stats)
    pred_diff = pred_a_norm - pred_b_norm
    true_diff = stats.norm_sibling(batch.target_a - batch.target_b, batch.cut)
    per_item, count = _per_item_mean(pred_diff, true_diff, mask)
    return _reduce_items(per_item, count, item_mask=batch.has_sibling)


def rollout_loss(
    model: nn.Module,
    batch: PairBatch,
    stats: NormStats,
    horizons: tuple[int, ...] = (2, 4),
) -> tuple[TermValue, dict[int, TermValue]]:
    """L_roll과 horizon별 값. 범위를 넘는 horizon은 정의되지 않은 것으로 남깁니다."""
    n_blocks = batch.input_a.n_blocks
    max_h = max(horizons)
    result_a = rollout(model, batch.input_a, stats, max_h)
    result_b = rollout(model, batch.input_b, stats, max_h)
    per_horizon: dict[int, TermValue] = {}
    values, counts = [], []
    for horizon in horizons:
        depth = batch.cut + horizon
        if depth > n_blocks:
            per_horizon[horizon] = TermValue(value=batch.target_a.sum() * 0.0, count=0)
            continue
        if not bool(stats.rollout_active[depth]):
            per_horizon[horizon] = TermValue(value=batch.target_a.sum() * 0.0, count=0)
            continue
        mask = batch.valid.view(batch.valid.shape[0], -1, 1)
        terms = []
        for result, truth in (
            (result_a, batch.future_state_diff_a),
            (result_b, batch.future_state_diff_b),
        ):
            pred = result.state_diff_at(depth)
            if pred is None:
                continue
            pred_n = stats.norm_rollout(pred, depth)
            true_n = stats.norm_rollout(truth[:, depth], depth)
            per_item, count = _per_item_mean(pred_n, true_n, mask)
            item_mask = None if result is result_a else batch.has_sibling
            term = _reduce_items(per_item, count, item_mask=item_mask)
            if term.defined:
                terms.append(term)
        if not terms:
            per_horizon[horizon] = TermValue(value=batch.target_a.sum() * 0.0, count=0)
            continue
        value = sum(t.value for t in terms) / len(terms)
        per_horizon[horizon] = TermValue(value=value, count=terms[0].count)
        values.append(value)
        counts.append(terms[0].count)
    if not values:
        return TermValue(value=batch.target_a.sum() * 0.0, count=0), per_horizon
    return TermValue(value=sum(values) / len(values), count=max(counts)), per_horizon


@dataclass
class LossBreakdown:
    total: Tensor
    next_update: TermValue
    within: TermValue
    roll: TermValue
    per_horizon: dict[int, TermValue]

    def as_dict(self) -> dict[str, float]:
        out = {
            "total": float(self.total),
            "L_next": self.next_update.item(),
            "L_within": self.within.item(),
            "L_roll": self.roll.item(),
        }
        for horizon, term in self.per_horizon.items():
            out[f"L_roll_h{horizon}"] = term.item()
        return out


def compute_loss(
    model: nn.Module,
    batch: PairBatch,
    stats: NormStats,
    w_next: float = 1.0,
    w_within: float = 1.0,
    w_roll: float = 0.25,
    horizons: tuple[int, ...] = (2, 4),
) -> LossBreakdown:
    """전체 loss. rollout 항의 gradient도 유지합니다."""
    pred_a = model(batch.input_a, stats)
    pred_b = model(batch.input_b, stats)
    term_next = next_update_loss(pred_a, pred_b, batch, stats)
    term_within = within_original_loss(pred_a, pred_b, batch, stats)
    term_roll, per_horizon = rollout_loss(model, batch, stats, horizons)
    total = term_next.value * 0.0
    if term_next.defined:
        total = total + w_next * term_next.value
    if term_within.defined:
        total = total + w_within * term_within.value
    if term_roll.defined:
        total = total + w_roll * term_roll.value
    return LossBreakdown(
        total=total,
        next_update=term_next,
        within=term_within,
        roll=term_roll,
        per_horizon=per_horizon,
    )
