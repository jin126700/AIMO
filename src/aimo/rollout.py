"""Horizon 2/4 rollout.

순서는 항상 normalize -> predict -> inverse normalize -> raw recurrence 입니다.
정규화 좌표에 raw update를 더하지 않습니다.

    U_hat_variant[d]   = U_original[d] + V_hat[d]                       (raw)
    H_hat_variant[d+1] = H_variant_or_pred[d] + sum_c U_hat_variant[d,c] (raw)

최초 cut 이후에는 실제 variant future를 다시 쓰지 않고, 직전 step의 예측값을 그대로
다음 prefix에 붙입니다. layer 범위를 넘는 step은 결과에서 제외되므로 loss와
evaluation에서 자동으로 빠집니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .data import NormStats, PredictorInput


@dataclass
class RolloutStep:
    depth: int  # 이 step이 예측한 block index d
    v_hat_norm: Tensor  # [B, P, 2, H] normalized V_hat[d]
    v_hat_raw: Tensor  # [B, P, 2, H] raw V_hat[d]
    state_hat_next: Tensor  # [B, P, H] raw H_hat_variant[d+1]
    state_diff_hat_next: Tensor  # [B, P, H] raw D_hat[d+1]


@dataclass
class RolloutResult:
    cut: int
    steps: list[RolloutStep]

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def state_diff_at(self, depth: int) -> Tensor | None:
        """depth(= d+1)에서의 D_hat. 범위를 넘으면 None입니다."""
        for step in self.steps:
            if step.depth + 1 == depth:
                return step.state_diff_hat_next
        return None


def rollout(
    model: nn.Module,
    inp: PredictorInput,
    stats: NormStats,
    horizon: int,
) -> RolloutResult:
    """cut에서 시작해 최대 horizon step을 예측합니다. gradient는 유지합니다."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    n_blocks = inp.n_blocks
    cut = inp.cut
    # prefix는 관측된 실제 variant 값에서 시작합니다. 이후에는 예측값만 붙입니다.
    state_prefix = inp.var_state_prefix
    upd_prefix = inp.var_updates_prefix
    steps: list[RolloutStep] = []

    for offset in range(horizon):
        depth = cut + offset
        if depth >= n_blocks:
            break  # layer 범위를 넘는 step은 만들지 않습니다.
        step_input = PredictorInput(
            cut=depth,
            orig_state=inp.orig_state,
            orig_updates=inp.orig_updates,
            var_state_prefix=state_prefix,
            var_updates_prefix=upd_prefix,
            valid=inp.valid,
            relative_positions=inp.relative_positions,
            token_offsets=inp.token_offsets,
        )
        v_hat_norm = model(step_input, stats)  # normalized space
        v_hat_raw = stats.denorm_target(v_hat_norm, depth)  # inverse normalize
        u_hat = inp.orig_updates[:, depth] + v_hat_raw  # raw update
        state_next = state_prefix[:, depth] + u_hat.sum(dim=-2)  # raw recurrence
        diff_next = state_next - inp.orig_state[:, depth + 1]
        steps.append(
            RolloutStep(
                depth=depth,
                v_hat_norm=v_hat_norm,
                v_hat_raw=v_hat_raw,
                state_hat_next=state_next,
                state_diff_hat_next=diff_next,
            )
        )
        # 예측된 state/update를 다음 prefix에 붙입니다 (실제 future 재사용 없음).
        state_prefix = torch.cat([state_prefix, state_next.unsqueeze(1)], dim=1)
        upd_prefix = torch.cat([upd_prefix, u_hat.unsqueeze(1)], dim=1)

    return RolloutResult(cut=cut, steps=steps)
