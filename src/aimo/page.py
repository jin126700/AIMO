"""Page 데이터 계약: shape, indexing, residual identity, pair difference.

Page는 고정 LLM 한 prompt의 내부 흐름을 landmark 위치에서만 잘라낸 관측입니다.
Batch dimension을 제외하면 다음 두 tensor가 계약의 중심입니다.

    state:   [L + 1, P, H]      # block 경계의 residual stream
    updates: [L, P, 2, H]       # block별 stream 기여

stream index는 고정입니다: 0 = Mixer (self-attention), 1 = FFN.
Block index d는 0-based이고 residual identity는

    state[d + 1] = state[d] + updates[d, :, 0] + updates[d, :, 1]

state[0]은 첫 block의 실제 input입니다. Final LayerNorm 이후의 hidden state는
마지막 block output이 아니므로 state[L]과 혼동하지 않습니다.

H는 원래 model hidden size를 그대로 보존합니다. random projection이나 learned
target encoder로 축소하지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

STREAM_MIXER = 0
STREAM_FFN = 1
N_STREAMS = 2

# 문제 본문의 실제 token landmark 16개 + canonical final prompt token 1개.
N_BODY_LANDMARKS = 16
N_LANDMARKS = N_BODY_LANDMARKS + 1

RESIDUAL_TOL = 1e-4


@dataclass
class Page:
    """한 prompt에 대한 landmark 관측.

    valid[p]가 False인 landmark는 padding 또는 중복 landmark이며 loss/metric에서
    제외됩니다. 원본과 변형의 같은 landmark 순번이 같은 수학 개념이라고 가정하지
    않습니다. valid는 depth와 무관한 [P] mask입니다: 한 prompt 안의 token 위치는
    모든 layer에서 고정입니다.
    """

    state: Tensor  # [L+1, P, H] float32
    updates: Tensor  # [L, P, 2, H] float32
    valid: Tensor  # [P] bool
    token_offsets: Tensor  # [P] int64, rendered prompt 안의 절대 token index
    relative_positions: Tensor  # [P] float32, token_offset / max(prompt_len - 1, 1)
    original_id: str
    variant_id: str
    is_identity: bool = False
    provenance: dict = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return int(self.updates.shape[0])

    @property
    def n_landmarks(self) -> int:
        return int(self.state.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.state.shape[-1])

    def residual_identity_error(self) -> float:
        """max |state[d+1] - (state[d] + sum_c updates[d, :, c])| over valid landmarks."""
        lhs = self.state[1:]  # [L, P, H]
        rhs = self.state[:-1] + self.updates.sum(dim=2)
        err = (lhs - rhs).abs()
        mask = self.valid.view(1, -1, 1)
        return float(err.masked_select(mask.expand_as(err)).max()) if bool(mask.any()) else 0.0

    def validate(self, tol: float = RESIDUAL_TOL) -> None:
        check_page(self, tol=tol)


def check_page(page: Page, tol: float = RESIDUAL_TOL) -> None:
    """shape / dtype / mask / residual identity 계약을 검증합니다."""
    if page.state.ndim != 3:
        raise ValueError(f"state must be [L+1, P, H], got {tuple(page.state.shape)}")
    if page.updates.ndim != 4:
        raise ValueError(f"updates must be [L, P, 2, H], got {tuple(page.updates.shape)}")
    n_state, n_p, hidden = page.state.shape
    n_blocks, n_p_u, n_streams, hidden_u = page.updates.shape
    if n_state != n_blocks + 1:
        raise ValueError(f"state depth {n_state} must equal updates depth {n_blocks} + 1")
    if n_p_u != n_p:
        raise ValueError(f"landmark count mismatch: state {n_p} vs updates {n_p_u}")
    if n_streams != N_STREAMS:
        raise ValueError(f"expected {N_STREAMS} streams (Mixer, FFN), got {n_streams}")
    if hidden_u != hidden:
        raise ValueError(f"hidden size mismatch: state {hidden} vs updates {hidden_u}")
    for name, tensor in (("state", page.state), ("updates", page.updates)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
    if page.valid.dtype != torch.bool or page.valid.shape != (n_p,):
        raise ValueError(
            f"valid must be bool [{n_p}], got {page.valid.dtype} {tuple(page.valid.shape)}"
        )
    if not bool(page.valid.any()):
        raise ValueError("page has no valid landmark")
    if page.token_offsets.shape != (n_p,) or page.token_offsets.dtype != torch.int64:
        raise ValueError("token_offsets must be int64 [P]")
    if page.relative_positions.shape != (n_p,) or page.relative_positions.dtype != torch.float32:
        raise ValueError("relative_positions must be float32 [P]")
    err = page.residual_identity_error()
    if err > tol:
        raise ValueError(f"residual identity violated: max abs error {err:.3e} > {tol:.3e}")


def state_diff(original: Page, variant: Page) -> Tensor:
    """D[d] = state_variant[d] - state_original[d], shape [L+1, P, H]."""
    _check_pair_shapes(original, variant)
    return variant.state - original.state


def update_diff(original: Page, variant: Page) -> Tensor:
    """V[d, c] = updates_variant[d, c] - updates_original[d, c], shape [L, P, 2, H]."""
    _check_pair_shapes(original, variant)
    return variant.updates - original.updates


def common_valid(original: Page, variant: Page) -> Tensor:
    """원본과 변형 모두에서 유효한 landmark mask, shape [P]."""
    _check_pair_shapes(original, variant)
    return original.valid & variant.valid


def pair_residual_identity_error(state_d: Tensor, update_v: Tensor, valid: Tensor) -> float:
    """D[d+1] = D[d] + sum_c V[d, c] 를 valid landmark에서 확인합니다."""
    lhs = state_d[1:]
    rhs = state_d[:-1] + update_v.sum(dim=2)
    err = (lhs - rhs).abs()
    mask = valid.view(1, -1, 1).expand_as(err)
    return float(err.masked_select(mask).max()) if bool(mask.any()) else 0.0


def _check_pair_shapes(original: Page, variant: Page) -> None:
    if original.state.shape != variant.state.shape:
        raise ValueError(
            f"state shape mismatch: {tuple(original.state.shape)} vs {tuple(variant.state.shape)}"
        )
    if original.updates.shape != variant.updates.shape:
        raise ValueError("updates shape mismatch between original and variant")
    if original.original_id != variant.original_id:
        raise ValueError(
            f"pair must share original_id: {original.original_id!r} vs {variant.original_id!r}"
        )


def provenance_hash(payload: dict) -> str:
    """model/tokenizer/config 등의 provenance dict에 대한 안정적인 short hash."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def save_pages(pages: list[Page], path: str | Path) -> Path:
    """Page 목록을 npz 하나로 저장합니다 (CPU float32)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    meta = []
    for i, page in enumerate(pages):
        arrays[f"state_{i}"] = page.state.detach().cpu().numpy()
        arrays[f"updates_{i}"] = page.updates.detach().cpu().numpy()
        arrays[f"valid_{i}"] = page.valid.detach().cpu().numpy()
        arrays[f"offsets_{i}"] = page.token_offsets.detach().cpu().numpy()
        arrays[f"relpos_{i}"] = page.relative_positions.detach().cpu().numpy()
        meta.append(
            {
                "original_id": page.original_id,
                "variant_id": page.variant_id,
                "is_identity": page.is_identity,
                "provenance": page.provenance,
            }
        )
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    return path


def load_pages(path: str | Path) -> list[Page]:
    """save_pages로 저장한 npz를 다시 읽습니다."""
    with np.load(Path(path), allow_pickle=False) as handle:
        meta = json.loads(str(handle["meta"]))
        pages = []
        for i, entry in enumerate(meta):
            pages.append(
                Page(
                    state=torch.from_numpy(handle[f"state_{i}"]).float(),
                    updates=torch.from_numpy(handle[f"updates_{i}"]).float(),
                    valid=torch.from_numpy(handle[f"valid_{i}"]).bool(),
                    token_offsets=torch.from_numpy(handle[f"offsets_{i}"]).long(),
                    relative_positions=torch.from_numpy(handle[f"relpos_{i}"]).float(),
                    original_id=entry["original_id"],
                    variant_id=entry["variant_id"],
                    is_identity=bool(entry["is_identity"]),
                    provenance=entry["provenance"],
                )
            )
    return pages
