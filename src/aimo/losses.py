"""Loss 항: Behavior supervision과 Flow auxiliary.

    L        = L_behavior + 0.1 * L_flow

    L_behavior = BCE(robust) + Huber_0.1(signed pair drop) [+ Huber_0.1(panel-only max drop)]
    L_flow     = L_next + L_within + 0.25 * L_roll

    L_next   = normalized MSE(V_hat, V)
    L_within = normalized MSE(V_hat_a - V_hat_b, V_a - V_b)
    L_roll   = normalized MSE(D_hat_future, D_future), horizon 2/4 평균

모든 항은 원문(original) 단위로 먼저 평균한 뒤 원문끼리 동일 가중치로 평균합니다.
label이 없는 항은 mask-out하고 유효 count를 함께 기록합니다. label 0은 실제 label이므로
missing으로 오인하지 않습니다. pair drop과 max drop이 같은 counts에서 파생되면 이중
감독이 되므로 max-drop loss는 기본적으로 꺼져 있습니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .data import NormStats, PairBatch, PanelBatch
from .rollout import rollout

# Huber loss의 delta. pair drop과 panel max-drop에 공통으로 씁니다.
HUBER_DELTA = 0.1
# joint objective에서 flow auxiliary의 가중치.
FLOW_WEIGHT = 0.1


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


# --------------------------------------------------------------------------------------
# Behavior supervision
# --------------------------------------------------------------------------------------


@dataclass
class BehaviorLossBreakdown:
    """behavior 항별 값과 실제로 기여한 원문/pair 수."""

    total: Tensor
    robust: TermValue
    pair_drop: TermValue
    max_drop: TermValue
    n_pairs_with_drop: int = 0
    n_panels_with_robust: int = 0
    n_panels_with_max_drop: int = 0

    @property
    def any_supervision(self) -> bool:
        return self.robust.defined or self.pair_drop.defined or self.max_drop.defined

    def as_dict(self) -> dict[str, float]:
        return {
            "behavior_total": float(self.total),
            "L_robust": self.robust.item(),
            "L_pair_drop": self.pair_drop.item(),
            "L_max_drop": self.max_drop.item(),
            "n_pairs_with_drop": float(self.n_pairs_with_drop),
            "n_panels_with_robust": float(self.n_panels_with_robust),
            "n_panels_with_max_drop": float(self.n_panels_with_max_drop),
        }


def _panel_mean(values: Tensor, mask: Tensor) -> TermValue:
    """[B, M] 값을 panel 안에서 평균한 뒤 panel끼리 동일 가중치로 평균합니다."""
    weights = mask.to(values.dtype)
    per_panel_count = weights.sum(dim=1)
    per_panel = (values * weights).sum(dim=1) / per_panel_count.clamp_min(1.0)
    keep = per_panel_count > 0
    n = int(keep.sum())
    if n == 0:
        return TermValue(value=values.sum() * 0.0, count=0)
    return TermValue(value=(per_panel * keep.to(values.dtype)).sum() / n, count=n)


def _scalar_mean(values: Tensor, mask: Tensor) -> TermValue:
    """[B] 값을 mask된 panel에 대해 평균합니다."""
    n = int(mask.sum())
    if n == 0:
        return TermValue(value=values.sum() * 0.0, count=0)
    safe = torch.where(mask, values, torch.zeros_like(values))
    return TermValue(value=safe.sum() / n, count=n)


def behavior_loss(
    model,
    batch: PanelBatch,
    stats: NormStats,
    w_robust: float = 1.0,
    w_pair_drop: float = 1.0,
    w_max_drop: float = 1.0,
    use_max_drop: bool = False,
) -> BehaviorLossBreakdown:
    """L_behavior. 없는 label 항은 mask-out하고 유효 count를 기록합니다."""
    out = model.forward_behavior(batch.inputs, stats)
    panel = model.panel_outputs(out, batch.pair_panel, batch.pair_slot, batch.panel_mask)

    # pair signed drop: Huber(delta=0.1). NaN target은 mask 전에 0으로 치환합니다.
    drop_pred = torch.zeros_like(batch.drop_target)
    drop_pred = drop_pred.index_put(
        (batch.pair_panel, batch.pair_slot), out.pair_drop, accumulate=False
    )
    drop_target = torch.where(
        batch.drop_mask, batch.drop_target, torch.zeros_like(batch.drop_target)
    )
    drop_values = F.huber_loss(
        drop_pred, drop_target, reduction="none", delta=HUBER_DELTA
    )
    term_drop = _panel_mean(drop_values, batch.drop_mask)

    # original-panel robustness: BCE with logits. panel_valid가 아닌 row는 제외합니다.
    robust_mask = batch.robust_mask & panel.panel_valid
    logit = torch.where(panel.panel_valid, panel.robust_logit, torch.zeros_like(panel.robust_logit))
    robust_target = torch.where(
        robust_mask, batch.robust_target, torch.zeros_like(batch.robust_target)
    )
    robust_values = F.binary_cross_entropy_with_logits(
        logit, robust_target, reduction="none"
    )
    term_robust = _scalar_mean(robust_values, robust_mask)

    # panel max-drop: 독립적인 panel-only target이 있을 때만 loss로 씁니다.
    max_mask = batch.max_drop_mask & panel.panel_valid if use_max_drop else torch.zeros_like(
        batch.max_drop_mask
    )
    max_pred = torch.where(panel.panel_valid, panel.max_drop, torch.zeros_like(panel.max_drop))
    max_target = torch.where(
        max_mask, batch.max_drop_target, torch.zeros_like(batch.max_drop_target)
    )
    max_values = F.huber_loss(max_pred, max_target, reduction="none", delta=HUBER_DELTA)
    term_max = _scalar_mean(max_values, max_mask)

    total = term_drop.value * 0.0
    if term_robust.defined:
        total = total + w_robust * term_robust.value
    if term_drop.defined:
        total = total + w_pair_drop * term_drop.value
    if term_max.defined:
        total = total + w_max_drop * term_max.value
    return BehaviorLossBreakdown(
        total=total,
        robust=term_robust,
        pair_drop=term_drop,
        max_drop=term_max,
        n_pairs_with_drop=int(batch.drop_mask.sum()),
        n_panels_with_robust=int(robust_mask.sum()),
        n_panels_with_max_drop=int(max_mask.sum()),
    )


@dataclass
class JointLossBreakdown:
    """joint objective. flow가 없으면 flow 항은 None입니다."""

    total: Tensor
    behavior: BehaviorLossBreakdown | None
    flow: LossBreakdown | None
    flow_weight: float = FLOW_WEIGHT

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {"total": float(self.total), "flow_weight": self.flow_weight}
        if self.behavior is not None:
            out.update(self.behavior.as_dict())
        if self.flow is not None:
            out.update(self.flow.as_dict())
            out["flow_total"] = float(self.flow.total)
        return out


def joint_loss(
    behavior: BehaviorLossBreakdown | None,
    flow: LossBreakdown | None,
    flow_weight: float = FLOW_WEIGHT,
) -> JointLossBreakdown:
    """L = L_behavior + flow_weight * L_flow. 두 항의 gradient는 같은 core에 누적됩니다."""
    if behavior is None and flow is None:
        raise ValueError("joint_loss needs at least one of behavior / flow")
    parts = []
    if behavior is not None and behavior.any_supervision:
        parts.append(behavior.total)
    if flow is not None:
        parts.append(flow_weight * flow.total)
    if not parts:
        reference = behavior.total if behavior is not None else flow.total  # type: ignore[union-attr]
        total = reference * 0.0
    else:
        total = parts[0]
        for extra in parts[1:]:
            total = total + extra
    return JointLossBreakdown(
        total=total, behavior=behavior, flow=flow, flow_weight=flow_weight
    )
