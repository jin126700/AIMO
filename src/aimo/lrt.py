"""LRT-v1: Looped Relational Transport.

질문: original→variant relation의 일부 computation site를 보고 만든 저차원 relational
state `z_rel`이, **보지 않은** computation site의 relation을 설명할 수 있는가.

    Support relation  ->  z_rel  ->  Held-out relation

이 실험은 future prediction이 아닙니다. support에는 query보다 뒤의 macro가 들어갈 수
있으므로 정확한 이름은 **cross-macro relational transport**(cross-site reconstruction)이며
causal transport라고 쓰지 않습니다.

이 Stage에서는 correctness / pair-drop / max-drop / robust label / perturbation recipe를
전혀 쓰지 않습니다.

입출력 계약:

    encoder input   : A_U(ΔU_norm[g, p, c])  + site metadata      (support macro만)
    z_rel           : R^16
    decoder context : z_rel, A_S(O.state[g,p]), A_U(O.updates[g,p,c]), site metadata
    decoder output  : ΔU_hat[g, p, c] = B_c(a),  a in R^16

누적량 `V.state - O.state`는 encoder input으로 쓰지 않습니다. variant query update는
decoder input에 절대 들어가지 않습니다. original query update는 target이 아니므로
leakage가 아닙니다.

historical U4는 대략 `ΔU ≈ P z`로 original computation과 무관한 fixed transformation이었고,
LRT-v1은 같은 low-dimensional output subspace를 유지하면서 relation code / state 의존 /
site 의존을 허용합니다. 다만 full H-space arbitrary nonlinear reconstruction은 허용하지
않습니다 (decoder rank 16 native basis).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .macro_page import MacroPage, common_valid, macro_relation
from .model import SharedBlock

LRT_SCHEMA = "aimo-lrt-v1"

STREAM_MIXER = 0
STREAM_FFN = 1
N_STREAMS = 2

LANDMARK_ALL_COMMON = "all_common"
LANDMARK_FINAL_TOKEN = "final_token"
LANDMARK_MODES = (LANDMARK_ALL_COMMON, LANDMARK_FINAL_TOKEN)


def query_folds(n_macro: int) -> list[tuple[int, ...]]:
    """query fold는 연속된 macro pair입니다. G=8이면 [0,1] [2,3] [4,5] [6,7].

    나머지 macro는 support입니다. 각 pair에 대해 fold를 rotate합니다.
    """
    if n_macro < 4 or n_macro % 2 != 0:
        raise ValueError(f"n_macro must be an even number >= 4, got {n_macro}")
    return [(g, g + 1) for g in range(0, n_macro, 2)]


def landmark_mask(
    original: MacroPage, variant: MacroPage, mode: str = LANDMARK_ALL_COMMON
) -> Tensor:
    """evaluation mode에 맞는 landmark mask, shape [P].

    all_common  : common-valid landmark 전부 (primary train)
    final_token : canonical final prompt landmark 하나만 (mandatory diagnostic)
    """
    if mode not in LANDMARK_MODES:
        raise ValueError(f"landmark_mode must be one of {LANDMARK_MODES}, got {mode!r}")
    common = common_valid(original, variant)
    if mode == LANDMARK_ALL_COMMON:
        return common
    # canonical final prompt landmark = 마지막 landmark 순번.
    mask = torch.zeros_like(common)
    mask[-1] = common[-1]
    return mask


@dataclass
class MacroNormStats:
    """LRT normalization. **train originals만**으로 fit합니다.

    variant delta 분포를 먼저 보고 scale을 정하지 않습니다. validation/test Page 통계는
    쓰지 않습니다.
    """

    state_scale: Tensor  # [G+1]
    update_scale: Tensor  # [G, 2]
    floor: float = 1e-6
    source: str = "train_originals"
    n_originals: int = 0

    def _safe(self, scale: Tensor) -> Tensor:
        return scale.clamp_min(self.floor)

    def norm_state(self, state: Tensor) -> Tensor:
        """[..., G+1, P, H] / state_scale."""
        depth = state.shape[-3]
        return state / self._safe(self.state_scale[:depth]).view(-1, 1, 1)

    def norm_update(self, updates: Tensor) -> Tensor:
        """[..., G, P, 2, H] / update_scale."""
        depth = updates.shape[-4]
        return updates / self._safe(self.update_scale[:depth]).view(-1, 1, 2, 1)

    def to(self, device: torch.device) -> MacroNormStats:
        if self.state_scale.device == device:
            return self
        return MacroNormStats(
            state_scale=self.state_scale.to(device),
            update_scale=self.update_scale.to(device),
            floor=self.floor,
            source=self.source,
            n_originals=self.n_originals,
        )

    def state_dict(self) -> dict:
        return {
            "state_scale": self.state_scale.detach().cpu(),
            "update_scale": self.update_scale.detach().cpu(),
            "floor": self.floor,
            "source": self.source,
            "n_originals": self.n_originals,
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> MacroNormStats:
        return cls(
            state_scale=payload["state_scale"],
            update_scale=payload["update_scale"],
            floor=float(payload["floor"]),
            source=payload.get("source", "train_originals"),
            n_originals=int(payload.get("n_originals", 0)),
        )

    def hash(self) -> str:
        import hashlib

        blob = b"".join(
            tensor.detach().cpu().numpy().tobytes()
            for tensor in (self.state_scale, self.update_scale)
        )
        return hashlib.sha256(blob).hexdigest()[:16]


def fit_macro_norm_stats(
    originals: list[MacroPage], floor: float = 1e-6
) -> MacroNormStats:
    """train **originals**의 valid landmark에서 macro depth별 RMS를 구합니다."""
    if not originals:
        raise ValueError("macro normalization needs at least one train original")
    n_macro = originals[0].n_macro
    state_sq = torch.zeros(n_macro + 1, dtype=torch.float64)
    state_n = torch.zeros(n_macro + 1, dtype=torch.float64)
    upd_sq = torch.zeros(n_macro, 2, dtype=torch.float64)
    upd_n = torch.zeros(n_macro, 2, dtype=torch.float64)
    for page in originals:
        mask = page.valid
        state = page.state[:, mask].double()
        state_sq += state.pow(2).sum(dim=(1, 2))
        state_n += state.shape[1] * state.shape[2]
        updates = page.updates[:, mask].double()
        upd_sq += updates.pow(2).sum(dim=(1, 3))
        upd_n += updates.shape[1] * updates.shape[3]
    state_scale = (state_sq / state_n.clamp_min(1)).sqrt().float()
    update_scale = (upd_sq / upd_n.clamp_min(1)).sqrt().float()
    return MacroNormStats(
        state_scale=state_scale,
        update_scale=update_scale,
        floor=floor,
        n_originals=len(originals),
    )


@dataclass
class LRTParamReport:
    """component별 parameter 수. Macro8이 parameter를 자동으로 4배 줄인다고 쓰지 않습니다."""

    update_adapter: int = 0
    state_adapter: int = 0
    relation_embeddings: int = 0
    shared_block: int = 0
    relation_projection: int = 0
    coefficient_network: int = 0
    mixer_basis: int = 0
    ffn_basis: int = 0
    total: int = 0
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "update_adapter": self.update_adapter,
            "state_adapter": self.state_adapter,
            "relation_embeddings": self.relation_embeddings,
            "shared_block": self.shared_block,
            "relation_projection": self.relation_projection,
            "coefficient_network": self.coefficient_network,
            "mixer_basis": self.mixer_basis,
            "ffn_basis": self.ffn_basis,
            "total": self.total,
            "notes": dict(self.notes),
        }


@dataclass
class LRTOutput:
    """relation code와 query 예측."""

    z_rel: Tensor  # [B, relation_dim]
    delta_hat: Tensor  # [B, Q, P, 2, H] normalized 예측
    support_cells: int = 0


class SiteEmbedding(nn.Module):
    """macro depth / landmark / stream / relative token position metadata embedding.

    `nn.Embedding` 기본 초기화(std 1.0)는 adapter를 지난 relation 신호(std ~0.2)를 약 9배로
    압도해 LayerNorm 뒤에 relation 내용이 묻힙니다. 그래서 metadata는 작은 std로 초기화합니다.
    """

    INIT_STD = 0.02

    def __init__(self, n_macro: int, n_landmarks: int, d_model: int) -> None:
        super().__init__()
        self.depth = nn.Embedding(n_macro + 1, d_model)
        self.landmark = nn.Embedding(n_landmarks, d_model)
        self.stream = nn.Embedding(N_STREAMS, d_model)
        self.position = nn.Linear(1, d_model)
        for table in (self.depth, self.landmark, self.stream):
            nn.init.normal_(table.weight, mean=0.0, std=self.INIT_STD)

    def forward(
        self, depth: Tensor, landmark: Tensor, stream: Tensor, relative_position: Tensor
    ) -> Tensor:
        return (
            self.depth(depth)
            + self.landmark(landmark)
            + self.stream(stream)
            + self.position(relative_position.unsqueeze(-1))
        )


class LRTModel(nn.Module):
    """LRT-v1 encoder + low-rank native decoder.

    encoder는 shared pre-LN block 하나를 `n_loops`회 호출합니다 (같은 parameter 객체).
    decoder는 stream별 native basis `Linear(decoder_rank, H, bias=False)`로만 native
    space에 씁니다.
    """

    def __init__(
        self,
        hidden_size: int,
        n_macro: int,
        n_landmarks: int,
        *,
        adapter_dim: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        n_loops: int = 4,
        relation_dim: int = 16,
        decoder_rank: int = 16,
    ) -> None:
        super().__init__()
        if n_loops < 1:
            raise ValueError("n_loops must be >= 1")
        self.hidden_size = hidden_size
        self.n_macro = n_macro
        self.n_landmarks = n_landmarks
        self.adapter_dim = adapter_dim
        self.d_model = d_model
        self.n_loops = n_loops
        self.relation_dim = relation_dim
        self.decoder_rank = decoder_rank

        # ---- model-specific low-dimensional native adapters ----
        # Mixer와 FFN의 relation/original update는 **같은** A_U parameter를 공유하고,
        # stream 구분은 stream embedding이 담당합니다. cross-model에서는 이 H -> 32
        # adapter만 model-specific하게 바꾸면 됩니다.
        self.update_adapter = nn.Linear(hidden_size, adapter_dim)
        self.state_adapter = nn.Linear(hidden_size, adapter_dim)

        # ---- relation cell embedding ----
        self.cell_proj = nn.Linear(adapter_dim, d_model)
        self.site = SiteEmbedding(n_macro, n_landmarks, d_model)
        self.cell_norm = nn.LayerNorm(d_model)
        self.inject = nn.Linear(d_model, d_model, bias=False)

        # ---- shared looped block (동일 parameter 객체를 n_loops회 호출) ----
        self.block = SharedBlock(d_model, n_heads, ffn_dim, dropout)

        # ---- relation projection ----
        self.relation_norm = nn.LayerNorm(d_model)
        self.relation_proj = nn.Linear(d_model, relation_dim)

        # ---- coefficient generator (작은 network) ----
        coeff_in = relation_dim + 2 * adapter_dim + d_model
        self.coeff = nn.Sequential(
            nn.Linear(coeff_in, 64), nn.GELU(), nn.Linear(64, decoder_rank)
        )

        # ---- model-specific native bases (stream별) ----
        self.basis_mixer = nn.Linear(decoder_rank, hidden_size, bias=False)
        self.basis_ffn = nn.Linear(decoder_rank, hidden_size, bias=False)

    # -- parameter 보고 --------------------------------------------------------------
    def param_report(self) -> LRTParamReport:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        embeddings = count(self.cell_proj) + count(self.site) + count(self.cell_norm) + count(
            self.inject
        )
        return LRTParamReport(
            update_adapter=count(self.update_adapter),
            state_adapter=count(self.state_adapter),
            relation_embeddings=embeddings,
            shared_block=count(self.block),
            relation_projection=count(self.relation_norm) + count(self.relation_proj),
            coefficient_network=count(self.coeff),
            mixer_basis=count(self.basis_mixer),
            ffn_basis=count(self.basis_ffn),
            total=sum(p.numel() for p in self.parameters()),
            notes={
                "shared_block_called_times": self.n_loops,
                "relation_dim": self.relation_dim,
                "decoder_rank": self.decoder_rank,
                "adapter_dim": self.adapter_dim,
                "macro_effect": (
                    "macro는 depth 길이와 relation sequence 크기, attention compute를 줄이고 "
                    "native adapter가 IO parameter를 줄입니다. parameter가 자동으로 4배 "
                    "감소하는 것은 아닙니다."
                ),
            },
        )

    # -- site index helpers ----------------------------------------------------------
    def _site_index(
        self, macros: list[int], landmarks: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        """(depth, landmark, stream) index를 [n_macro * P * 2] 순서로 만듭니다."""
        n_p = int(landmarks.numel())
        depth = torch.tensor(macros, dtype=torch.long, device=device)
        depth = depth.repeat_interleave(n_p * N_STREAMS)
        landmark = landmarks.to(device).repeat_interleave(N_STREAMS).repeat(len(macros))
        stream = torch.arange(N_STREAMS, device=device).repeat(len(macros) * n_p)
        return depth, landmark, stream

    # -- encoder ---------------------------------------------------------------------
    def encode(
        self,
        delta_norm: Tensor,
        support_macros: list[int],
        landmarks: Tensor,
        relative_positions: Tensor,
        *,
        cell_mask: Tensor | None = None,
    ) -> tuple[Tensor, int]:
        """support relation cell에서 z_rel을 만듭니다.

        delta_norm : [B, G, P, 2, H] normalized ΔU (support macro만 읽습니다)
        landmarks  : [P_sel] 선택된 landmark index
        cell_mask  : [B, n_support * P_sel * 2] bool. consistency view용 dropout mask.
        """
        if not support_macros:
            raise ValueError("support macro set must not be empty")
        batch = delta_norm.shape[0]
        device = delta_norm.device
        selected = delta_norm[:, support_macros][:, :, landmarks]  # [B, Gs, P_sel, 2, H]
        cells = selected.reshape(batch, -1, self.hidden_size)  # [B, S, H]
        depth, landmark, stream = self._site_index(support_macros, landmarks, device)
        position = relative_positions[:, landmarks].repeat_interleave(N_STREAMS, dim=1)
        position = position.repeat(1, len(support_macros))

        x0 = self.cell_norm(
            self.cell_proj(self.update_adapter(cells))
            + self.site(depth, landmark, stream, position)
        )
        mask = (
            torch.ones(batch, cells.shape[1], dtype=torch.bool, device=device)
            if cell_mask is None
            else cell_mask
        )
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every support cell was masked out; keep at least one cell")
        # padding/dropout된 cell은 key에서 제외하고, 전부 막힌 row가 없도록 보장합니다.
        blocked = ~mask.unsqueeze(1).expand(-1, cells.shape[1], -1)
        attn_mask = (
            blocked.unsqueeze(1)
            .expand(-1, self.block.attn.num_heads, -1, -1)
            .reshape(-1, cells.shape[1], cells.shape[1])
        )
        injected = self.inject(x0)
        hidden = x0
        for _ in range(self.n_loops):
            hidden = self.block(hidden + injected, attn_mask)
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        z_rel = self.relation_proj(self.relation_norm(pooled))
        return z_rel, int(mask.sum())

    # -- decoder ---------------------------------------------------------------------
    def decode(
        self,
        z_rel: Tensor,
        original_state_norm: Tensor,
        original_update_norm: Tensor,
        query_macros: tuple[int, ...],
        landmarks: Tensor,
        relative_positions: Tensor,
    ) -> Tensor:
        """query site의 ΔU_hat을 예측합니다.

        original_state_norm  : [B, G+1, P, H]
        original_update_norm : [B, G, P, 2, H]
        반환                 : [B, Q, P_sel, 2, H]
        """
        batch = z_rel.shape[0]
        device = z_rel.device
        n_q, n_p = len(query_macros), int(landmarks.numel())
        q_index = torch.tensor(query_macros, dtype=torch.long, device=device)

        state = original_state_norm[:, q_index][:, :, landmarks]  # [B, Q, P_sel, H]
        update = original_update_norm[:, q_index][:, :, landmarks]  # [B, Q, P_sel, 2, H]
        state_code = self.state_adapter(state).unsqueeze(-2).expand(-1, -1, -1, N_STREAMS, -1)
        update_code = self.update_adapter(update)  # 같은 A_U를 공유합니다

        depth, landmark, stream = self._site_index(list(query_macros), landmarks, device)
        position = relative_positions[:, landmarks].repeat_interleave(N_STREAMS, dim=1)
        position = position.repeat(1, n_q)
        site = self.site(depth, landmark, stream, position)
        site = site.view(batch, n_q, n_p, N_STREAMS, self.d_model)

        relation = z_rel.view(batch, 1, 1, 1, -1).expand(-1, n_q, n_p, N_STREAMS, -1)
        features = torch.cat([relation, state_code, update_code, site], dim=-1)
        coefficients = self.coeff(features)  # [B, Q, P_sel, 2, decoder_rank]
        mixer = self.basis_mixer(coefficients[..., STREAM_MIXER, :])
        ffn = self.basis_ffn(coefficients[..., STREAM_FFN, :])
        return torch.stack([mixer, ffn], dim=-2)  # [B, Q, P_sel, 2, H]

    # -- forward ---------------------------------------------------------------------
    def forward(
        self,
        delta_norm: Tensor,
        original_state_norm: Tensor,
        original_update_norm: Tensor,
        support_macros: list[int],
        query_macros: tuple[int, ...],
        landmarks: Tensor,
        relative_positions: Tensor,
        *,
        cell_mask: Tensor | None = None,
    ) -> LRTOutput:
        z_rel, support_cells = self.encode(
            delta_norm, support_macros, landmarks, relative_positions, cell_mask=cell_mask
        )
        delta_hat = self.decode(
            z_rel,
            original_state_norm,
            original_update_norm,
            query_macros,
            landmarks,
            relative_positions,
        )
        return LRTOutput(z_rel=z_rel, delta_hat=delta_hat, support_cells=support_cells)


def support_macros_for(n_macro: int, query_macros: tuple[int, ...]) -> list[int]:
    """query fold를 제외한 나머지 macro가 support입니다."""
    return [g for g in range(n_macro) if g not in set(query_macros)]


def build_pair_tensors(
    original: MacroPage, variant: MacroPage, stats: MacroNormStats
) -> dict[str, Tensor]:
    """한 pair의 normalized tensor를 만듭니다 (batch dim 1).

    encoder는 ΔU만 봅니다. decoder는 original state/update만 context로 씁니다.
    """
    delta = macro_relation(original, variant)[None]
    return {
        "delta_norm": stats.norm_update(delta),
        "original_state_norm": stats.norm_state(original.state[None]),
        "original_update_norm": stats.norm_update(original.updates[None]),
        "relative_positions": original.relative_positions[None],
    }
