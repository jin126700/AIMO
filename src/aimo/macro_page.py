"""MacroPage-8: Fine Page에서 runtime에 만드는 derived view.

Fine Page가 source of truth입니다. 이 module은 Fine Page를 **읽기만** 하고 mutate하지
않으며, 기존 `save_pages` / `load_pages` / fingerprint / provenance / extractor 계약을
바꾸지 않습니다. 따라서 LRT 때문에 기존 Page artifact를 다시 추출할 필요가 없습니다.

좌표:

    S[g, p]          = fine.state[b_g, p]                     shape [G+1, P, H]
    U_macro[g, p, c] = sum_{l=b_g}^{b_{g+1}-1} fine.updates[l, p, c]   shape [G, P, 2, H]
    b_g              = floor(g * L / G),  g = 0..G

boundary는 strictly increasing이어야 하고 `G <= L`이어야 합니다. layer 수를 hard-code하지
않으므로 24 / 32 / 36 / 48 layer 모두 같은 G=8 coordinate를 씁니다.

macro residual identity는 source Page tolerance 안에서 유지됩니다.

    S[g+1] = S[g] + U_macro[g, :, 0] + U_macro[g, :, 1]

`path_energy`는 macro sum에서 생기는 cancellation을 보는 **diagnostic 전용**입니다.
LRT-v1의 encoder input / decoder / training loss에는 쓰지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .page import N_STREAMS, Page

MACRO_SCHEMA = "aimo-macro-page-v1"
DEFAULT_N_MACRO = 8
# derived view의 residual identity 허용 오차. source Page tolerance와 같은 수준입니다.
MACRO_RESIDUAL_TOL = 1e-3


def macro_boundaries(n_blocks: int, n_macro: int = DEFAULT_N_MACRO) -> list[int]:
    """b_g = floor(g * L / G). strictly increasing이고 b_0 = 0, b_G = L입니다."""
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
    if n_macro < 1:
        raise ValueError(f"n_macro must be >= 1, got {n_macro}")
    if n_macro > n_blocks:
        raise ValueError(
            f"n_macro={n_macro} must be <= n_blocks={n_blocks}; a macro stage cannot be empty"
        )
    bounds = [(g * n_blocks) // n_macro for g in range(n_macro + 1)]
    for left, right in zip(bounds[:-1], bounds[1:], strict=True):
        if right <= left:
            raise ValueError(f"macro boundaries are not strictly increasing: {bounds}")
    if bounds[0] != 0 or bounds[-1] != n_blocks:
        raise ValueError(f"macro boundaries must span [0, {n_blocks}]: {bounds}")
    return bounds


@dataclass
class MacroPage:
    """Fine Page의 G-stage derived view. source Page를 복제해 보관하지 않습니다."""

    state: Tensor  # [G+1, P, H]
    updates: Tensor  # [G, P, 2, H]
    valid: Tensor  # [P] bool (source Page와 동일)
    token_offsets: Tensor  # [P] int64
    relative_positions: Tensor  # [P] float32
    original_id: str
    variant_id: str
    boundaries: tuple[int, ...]
    n_blocks: int  # source Fine Page의 L
    is_identity: bool = False
    provenance: dict = field(default_factory=dict)
    # diagnostic 전용. encoder/decoder/loss에 쓰지 않습니다.
    path_energy: Tensor | None = None

    @property
    def n_macro(self) -> int:
        return int(self.updates.shape[0])

    @property
    def n_landmarks(self) -> int:
        return int(self.state.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.state.shape[-1])

    def residual_identity_error(self) -> float:
        """valid landmark에서 macro residual identity의 최대 절대 오차."""
        lhs = self.state[1:]
        rhs = self.state[:-1] + self.updates.sum(dim=2)
        error = (lhs - rhs).abs()
        mask = self.valid.view(1, -1, 1).expand_as(error)
        return float(error.masked_select(mask).max()) if bool(mask.any()) else 0.0

    def validate(self, tol: float = MACRO_RESIDUAL_TOL) -> None:
        if self.state.shape[0] != self.updates.shape[0] + 1:
            raise ValueError("macro state depth must equal macro update depth + 1")
        if self.updates.shape[2] != N_STREAMS:
            raise ValueError(f"expected {N_STREAMS} streams, got {self.updates.shape[2]}")
        error = self.residual_identity_error()
        if error > tol:
            raise ValueError(
                f"macro residual identity violated: max abs error {error:.3e} > {tol:.3e}"
            )

    def fingerprint(self) -> str:
        """source Page fingerprint와 macro schema를 함께 담는 안정적 fingerprint.

        같은 Fine Page + 같은 G이면 같은 값입니다.
        """
        payload = {
            "macro_schema": MACRO_SCHEMA,
            "n_macro": self.n_macro,
            "boundaries": list(self.boundaries),
            "source_page_fingerprint": self.provenance.get("source_page_fingerprint", ""),
            "n_blocks": self.n_blocks,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def path_energy(page: Page, boundaries: list[int]) -> Tensor:
    """diagnostic: path_energy[g, p, c] = sum_l ||fine.update[l, p, c]||_2, shape [G, P, 2].

    macro sum에서 서로 상쇄된 fine update의 크기를 보기 위한 값입니다. **LRT-v1의 학습
    경로에는 들어가지 않습니다.**
    """
    norms = page.updates.norm(dim=-1)  # [L, P, 2]
    pairs = zip(boundaries[:-1], boundaries[1:], strict=True)
    return torch.stack([norms[left:right].sum(dim=0) for left, right in pairs])


def to_macro_page(
    page: Page,
    n_macro: int = DEFAULT_N_MACRO,
    *,
    with_path_energy: bool = False,
) -> MacroPage:
    """Fine Page에서 MacroPage를 deterministic하게 만듭니다. source는 그대로 둡니다."""
    bounds = macro_boundaries(page.n_blocks, n_macro)
    state = page.state[bounds].clone()  # [G+1, P, H]
    updates = torch.stack(
        [
            page.updates[left:right].sum(dim=0)
            for left, right in zip(bounds[:-1], bounds[1:], strict=True)
        ]
    )  # [G, P, 2, H]
    provenance = {
        "macro_schema": MACRO_SCHEMA,
        "n_macro": n_macro,
        "boundaries": list(bounds),
        "source_page_fingerprint": page.content_fingerprint(),
        "source_n_blocks": page.n_blocks,
        # source provenance를 잃지 않도록 그대로 들고 갑니다.
        "source_provenance": dict(page.provenance or {}),
    }
    macro = MacroPage(
        state=state,
        updates=updates,
        valid=page.valid.clone(),
        token_offsets=page.token_offsets.clone(),
        relative_positions=page.relative_positions.clone(),
        original_id=page.original_id,
        variant_id=page.variant_id,
        boundaries=tuple(bounds),
        n_blocks=page.n_blocks,
        is_identity=page.is_identity,
        provenance=provenance,
        path_energy=path_energy(page, bounds) if with_path_energy else None,
    )
    macro.validate()
    return macro


def macro_relation(original: MacroPage, variant: MacroPage) -> Tensor:
    """ΔU[g, p, c] = variant.updates - original.updates, shape [G, P, 2, H].

    LRT relation encoder가 보는 primary activation은 이 ΔU뿐입니다. 누적량인
    `variant.state - original.state`는 encoder input으로 쓰지 않습니다 (held-out macro
    정보의 indirect trace가 될 수 있습니다).
    """
    if original.updates.shape != variant.updates.shape:
        raise ValueError(
            f"macro update shape mismatch: {tuple(original.updates.shape)} vs "
            f"{tuple(variant.updates.shape)}"
        )
    if original.original_id != variant.original_id:
        raise ValueError(
            f"pair must share original_id: {original.original_id!r} vs {variant.original_id!r}"
        )
    return variant.updates - original.updates


def common_valid(original: MacroPage, variant: MacroPage) -> Tensor:
    """original.valid & variant.valid, shape [P].

    ordinal landmark p가 original/variant에서 같은 수학 개념이라고 가정하지 않으므로
    공통으로 유효한 landmark만 씁니다.
    """
    return original.valid & variant.valid


def relation_energy(delta: Tensor, valid: Tensor) -> float:
    """valid landmark에서의 relation energy sum ||ΔU||². Fine/Macro 비교 diagnostic용."""
    mask = valid.view(1, -1, 1, 1).expand_as(delta)
    return float((delta.pow(2) * mask).sum())


def relation_path_energy(original: Page, variant: Page, boundaries: list[int]) -> Tensor:
    """Fine pair relation의 path length [G,P,2]; diagnostic 전용."""
    delta = variant.updates - original.updates
    return torch.stack([delta[a:b].norm(dim=-1).sum(dim=0)
                        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)])
