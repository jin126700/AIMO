"""Offline sensitivity loss (VJP capture). 제출 entry point에서는 import하지 않습니다.

고정 seed의 native-response probe `v_j`에 대해

    a_gj = ∂(v_j^T y) / ∂S_g
    L_sensitivity = Σ ||a_gj - a_gj B_g B_g^T||² / max(Σ ||a_gj||², eps)

`y`는 native response sketch [T, q]이고 `v_j^T y = Σ_t m_t v_j · y_t`입니다 (m = valid mask).
full Jacobian이나 H × H projector를 만들지 않고 VJP 한 번과 r차원 projection으로 계산합니다.
`a_gj`는 target model에서 온 teacher gradient이므로 detach하고, VJP는 `create_graph=False`로
만들어 second-order gradient를 만들지 않습니다. target model weight는 학습하지 않습니다.

hybrid Mixer의 backward 지원, inference backend와 gradient backend의 수치 일치는 실제 model
검증 항목입니다. CPU toy에서 통과해도 실제 model VJP 검증으로 보고하지 않습니다.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

SENSITIVITY_EPS = 1e-8


def probe_vectors(n_probes: int, dim: int, seed: int) -> Tensor:
    """고정 seed의 unit probe [J, q]. 문제나 split을 보고 바꾸지 않습니다."""
    generator = torch.Generator().manual_seed(10_007 + seed)
    probes = torch.randn(n_probes, dim, generator=generator)
    return probes / probes.norm(dim=-1, keepdim=True)


def response_vjp(
    stage_forward: Callable[[Tensor], Tensor],
    state: Tensor,
    probes: Tensor,
    mask: Tensor,
) -> Tensor:
    """a [J, B, T, H] = ∂(v_j^T y)/∂S. VJP만 쓰고 결과를 detach합니다.

    stage_forward(state [B,T,H]) -> y [B,T,q]. mask [B,T].
    """
    leaf = state.detach().float().requires_grad_(True)
    with torch.enable_grad():
        y = stage_forward(leaf)
        weights = mask.to(y.dtype).unsqueeze(-1)
        grads = []
        for j in range(probes.shape[0]):
            cotangent = weights * probes[j].to(y.device, y.dtype)
            (grad,) = torch.autograd.grad(
                y, leaf, grad_outputs=cotangent, retain_graph=j + 1 < probes.shape[0],
                create_graph=False,
            )
            grads.append(grad.detach())
    return torch.stack(grads)


def sensitivity_terms(vjps: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    """(uncaptured energy, total energy) for one stage and one problem.

    vjps [J, T, H] (detached), basis [H, r]. ||a - aBB^T||² = ||a||² - ||aB||².
    """
    total = vjps.pow(2).sum()
    captured = (vjps @ basis).pow(2).sum()
    return (total - captured).clamp_min(0.0), total


def sensitivity_loss(
    vjps_by_stage: list[Tensor], basis: Tensor, eps: float = SENSITIVITY_EPS
) -> Tensor:
    """문제 하나의 L_sensitivity. vjps_by_stage[g] [J, T, H], basis [G, H, r]."""
    num = torch.zeros((), device=basis.device)
    den = torch.zeros((), device=basis.device)
    for g, vjps in enumerate(vjps_by_stage):
        miss, total = sensitivity_terms(vjps.to(basis.device), basis[g])
        num, den = num + miss, den + total
    return num / den.clamp_min(eps)


def directional_fd_check(
    stage_forward: Callable[[Tensor], Tensor],
    state: Tensor,
    probe: Tensor,
    mask: Tensor,
    direction: Tensor,
    step: float = 1e-3,
) -> dict:
    """VJP와 중앙 차분 directional derivative를 비교합니다 (backend agreement audit)."""
    vjp = response_vjp(stage_forward, state, probe.unsqueeze(0), mask)[0]
    analytic = float((vjp * direction).sum())
    weights = mask.float().unsqueeze(-1)
    with torch.no_grad():
        plus = (stage_forward(state + step * direction) * weights * probe).sum()
        minus = (stage_forward(state - step * direction) * weights * probe).sum()
    numeric = float((plus - minus) / (2 * step))
    return {
        "analytic": analytic,
        "numeric": numeric,
        "relative_error": abs(analytic - numeric) / max(abs(numeric), 1e-12),
    }
