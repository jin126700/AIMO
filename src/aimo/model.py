"""확정 Looped Transformer predictor와 비교군 model들.

구조는 다음 한 줄로 요약됩니다.

    native vectors
      -> layer 간 공유하는 작은 learned input embedding
      -> shared pre-LN Transformer block 반복 (동일 parameter 객체 4회 호출)
      -> layer 간 공유하는 native-space readout

LLM의 block index d와 predictor의 loop index k는 서로 다른 개념입니다. loop 하나가
LLM layer 하나를 재현한다고 주장하지 않습니다. 변화 문맥은 단일 z bottleneck이
아니라 sequence hidden states에 유지합니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .data import NormStats, PredictorInput

KIND_REF = 0
KIND_OBS = 1
KIND_QUERY = 2

STREAM_NONE = 0
STREAM_MIXER_TOKEN = 1
STREAM_FFN_TOKEN = 2


@dataclass
class ParamReport:
    total: int
    core: int
    input_output: int

    def as_dict(self) -> dict[str, int]:
        return {"total": self.total, "core": self.core, "input_output": self.input_output}


# --------------------------------------------------------------------------------------
# Sequence layout과 attention mask
# --------------------------------------------------------------------------------------


@dataclass
class SequenceLayout:
    """cell 배치와 masking에 필요한 index 정보.

    cell 순서: [reference depth 0..L] + [observed variant depth 0..cut] + [query 2P].
    같은 depth의 landmarks는 하나의 group으로 취급되어 mask row/column 패턴을
    공유합니다.
    """

    kind: Tensor  # [S] int64
    depth: Tensor  # [S] int64
    landmark: Tensor  # [S] int64
    stream: Tensor  # [S] int64, query cell만 mixer/ffn 구분
    query_index: Tensor  # [2P] int64, query cell의 sequence 위치
    n_landmarks: int
    cut: int


def build_layout(
    n_blocks: int, n_landmarks: int, cut: int, use_variant_prefix: bool, device: torch.device
) -> SequenceLayout:
    kinds, depths, landmarks, streams = [], [], [], []

    def add(kind: int, depth: int, stream: int) -> None:
        for p in range(n_landmarks):
            kinds.append(kind)
            depths.append(depth)
            landmarks.append(p)
            streams.append(stream)

    for r in range(n_blocks + 1):  # reference: original state 0..L, update r-1
        add(KIND_REF, r, STREAM_NONE)
    if use_variant_prefix:
        for r in range(cut + 1):  # observed variant: state 0..cut, update r-1
            add(KIND_OBS, r, STREAM_NONE)
    query_start = len(kinds)
    add(KIND_QUERY, cut, STREAM_MIXER_TOKEN)
    add(KIND_QUERY, cut, STREAM_FFN_TOKEN)
    query_index = torch.arange(query_start, len(kinds), device=device)
    to = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)  # noqa: E731
    return SequenceLayout(
        kind=to(kinds),
        depth=to(depths),
        landmark=to(landmarks),
        stream=to(streams),
        query_index=query_index,
        n_landmarks=n_landmarks,
        cut=cut,
    )


def build_attention_mask(layout: SequenceLayout, valid: Tensor) -> Tensor:
    """attend 가능 여부 mask. 반환값은 True가 '차단'인 bool [B, S, S]입니다.

    규칙:
      reference  -> reference만 읽습니다 (variant prefix를 전혀 읽지 않습니다).
      observed r -> reference 전체와 observed depth <= r.
      query      -> reference 전체와 observed depth <= cut.
      reference/observed는 query를 읽지 않습니다.
    padding landmark는 key에서 제외하고, 전부 차단된 row는 self를 열어 NaN을 막습니다.
    """
    kind, depth = layout.kind, layout.depth
    n_seq = kind.shape[0]
    kind_i = kind.view(-1, 1)
    kind_j = kind.view(1, -1)
    depth_i = depth.view(-1, 1)
    depth_j = depth.view(1, -1)

    j_is_ref = kind_j == KIND_REF
    j_is_obs = kind_j == KIND_OBS
    allow_ref_row = j_is_ref.expand(n_seq, n_seq)
    allow_obs_row = j_is_ref | (j_is_obs & (depth_j <= depth_i))
    allow_query_row = j_is_ref | (j_is_obs & (depth_j <= layout.cut))

    allowed = torch.where(kind_i == KIND_REF, allow_ref_row, allow_obs_row)
    allowed = torch.where(kind_i == KIND_QUERY, allow_query_row, allowed)  # [S, S]

    batch = valid.shape[0]
    allowed = allowed.unsqueeze(0).expand(batch, n_seq, n_seq).clone()
    key_valid = valid[:, layout.landmark]  # [B, S]
    allowed &= key_valid.unsqueeze(1)

    empty_rows = ~allowed.any(dim=-1)
    if bool(empty_rows.any()):
        eye = torch.eye(n_seq, dtype=torch.bool, device=allowed.device).unsqueeze(0)
        allowed |= empty_rows.unsqueeze(-1) & eye
    return ~allowed


# --------------------------------------------------------------------------------------
# Shared pre-LN block
# --------------------------------------------------------------------------------------


class SharedBlock(nn.Module):
    """pre-LN Transformer block 한 개. loop마다 같은 객체를 다시 호출합니다."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        normed = self.ln_attn(x)
        attended, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.ln_ffn(x)))


# --------------------------------------------------------------------------------------
# Looped predictor
# --------------------------------------------------------------------------------------


class LoopedPredictor(nn.Module):
    """cut d에서 V_hat[d] = [P, 2, H]를 normalized space로 예측합니다.

    Input embedding과 readout은 모든 depth가 공유하는 단일 module입니다. 각 loop
    앞에서 고정 observation embedding x0를 다시 주입합니다.

        x_0     = emb_norm(embed(observations))
        x_{k+1} = Block(x_k + inject(x_0))          k = 0 .. n_loops-1
        V_hat   = readout(readout_norm(x_K))[query cells]

    inject는 bias 없는 shared Linear 하나이며 모든 loop에서 동일한 parameter를 씁니다.
    Adaptive halting이나 per-loop 전용 module은 없습니다.
    """

    def __init__(
        self,
        hidden_size: int,
        n_blocks: int,
        n_landmarks: int,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        n_loops: int = 4,
        tied: bool = True,
        use_variant_prefix: bool = True,
    ) -> None:
        super().__init__()
        if n_loops < 1:
            raise ValueError("n_loops must be >= 1")
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_landmarks = n_landmarks
        self.d_model = d_model
        self.n_loops = n_loops
        self.tied = tied
        self.use_variant_prefix = use_variant_prefix

        # ---- input embedding (layer 간 공유) ----
        # cell feature = [state, update_mixer, update_ffn] 의 native vector 연결.
        self.value_proj = nn.Linear(3 * hidden_size, d_model)
        self.kind_emb = nn.Embedding(3, d_model)
        self.stream_emb = nn.Embedding(3, d_model)
        self.depth_emb = nn.Embedding(n_blocks + 1, d_model)
        self.landmark_emb = nn.Embedding(n_landmarks, d_model)
        # relative position과 cut까지의 상대 depth는 연속값으로 넣습니다.
        self.scalar_proj = nn.Linear(3, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.inject = nn.Linear(d_model, d_model, bias=False)

        # ---- core ----
        if tied:
            block = SharedBlock(d_model, n_heads, ffn_dim, dropout)
            self.blocks = nn.ModuleList([block])  # 동일 parameter 객체를 n_loops회 호출
        else:
            self.blocks = nn.ModuleList(
                [SharedBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_loops)]
            )

        # ---- readout (layer 간 공유, native H-space) ----
        self.readout_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, hidden_size)

    # -- parameter 보고 --------------------------------------------------------------
    def param_report(self) -> ParamReport:
        core = sum(p.numel() for p in self.blocks.parameters())
        io_modules = (
            self.value_proj,
            self.kind_emb,
            self.stream_emb,
            self.depth_emb,
            self.landmark_emb,
            self.scalar_proj,
            self.emb_norm,
            self.inject,
            self.readout_norm,
            self.readout,
        )
        io = sum(p.numel() for m in io_modules for p in m.parameters())
        total = sum(p.numel() for p in self.parameters())
        return ParamReport(total=total, core=core, input_output=io)

    # -- cell feature 구성 -----------------------------------------------------------
    def _cell_features(
        self, inp: PredictorInput, stats: NormStats, layout: SequenceLayout
    ) -> Tensor:
        """[B, S, 3H] cell feature.

        indexing 규칙: depth r cell에는 state[r]과 '이미 완료된' update[r-1]만 넣습니다.
        따라서 query target U[d]가 depth d cell에 들어가는 일은 없습니다.
        observed cell은 var_updates_prefix[0:cut]만 소비합니다.
        """
        batch = inp.batch_size
        hidden = self.hidden_size
        n_p = self.n_landmarks
        device = inp.orig_state.device

        orig_state = stats.norm_state(inp.orig_state)  # [B, L+1, P, H]
        orig_upd = stats.norm_update(inp.orig_updates)  # [B, L, P, 2, H]
        zeros_upd = torch.zeros(batch, 1, n_p, 2, hidden, device=device)
        # depth r의 '완료된 update'는 r-1. r=0은 0으로 채웁니다.
        orig_prev = torch.cat([zeros_upd, orig_upd], dim=1)  # [B, L+1, P, 2, H]
        ref_feat = torch.cat(
            [orig_state, orig_prev[..., 0, :], orig_prev[..., 1, :]], dim=-1
        )  # [B, L+1, P, 3H]

        pieces = [ref_feat]
        if self.use_variant_prefix:
            var_state = stats.norm_state(inp.var_state_prefix)  # [B, cut+1, P, H]
            if inp.cut > 0:
                var_upd = stats.norm_update(inp.var_updates_prefix)  # [B, cut, P, 2, H]
                var_prev = torch.cat([zeros_upd, var_upd], dim=1)
            else:
                var_prev = zeros_upd
            obs_feat = torch.cat(
                [var_state, var_prev[..., 0, :], var_prev[..., 1, :]], dim=-1
            )  # [B, cut+1, P, 3H]
            pieces.append(obs_feat)
        # query cell에는 activation을 넣지 않습니다: 위치와 역할 정보만 사용합니다.
        pieces.append(torch.zeros(batch, 2, n_p, 3 * hidden, device=device))

        flat = [piece.reshape(batch, -1, 3 * hidden) for piece in pieces]
        features = torch.cat(flat, dim=1)  # [B, S, 3H]
        if features.shape[1] != layout.kind.shape[0]:
            raise RuntimeError(
                f"sequence length mismatch: features {features.shape[1]} "
                f"vs layout {layout.kind.shape[0]}"
            )
        # padding landmark의 값은 0으로 지웁니다. key mask와 이중으로 차단합니다.
        cell_valid = inp.valid[:, layout.landmark].unsqueeze(-1)  # [B, S, 1]
        return features * cell_valid

    def _embed(self, features: Tensor, inp: PredictorInput, layout: SequenceLayout) -> Tensor:
        batch = features.shape[0]
        depth = layout.depth
        rel_cut = (depth.float() - float(layout.cut)) / max(self.n_blocks, 1)
        at_cut = (depth == layout.cut).float()
        rel_pos = inp.relative_positions[:, layout.landmark]  # [B, S]
        scalars = torch.stack(
            [
                rel_cut.view(1, -1).expand(batch, -1),
                at_cut.view(1, -1).expand(batch, -1),
                rel_pos,
            ],
            dim=-1,
        )  # [B, S, 3]
        emb = (
            self.value_proj(features)
            + self.kind_emb(layout.kind)
            + self.stream_emb(layout.stream)
            + self.depth_emb(depth.clamp_max(self.n_blocks))
            + self.landmark_emb(layout.landmark)
            + self.scalar_proj(scalars)
        )
        return self.emb_norm(emb)

    def forward(self, inp: PredictorInput, stats: NormStats) -> Tensor:
        """normalized V_hat[cut], shape [B, P, 2, H]."""
        if inp.n_blocks != self.n_blocks:
            raise ValueError(f"expected {self.n_blocks} blocks, got {inp.n_blocks}")
        device = inp.orig_state.device
        layout = build_layout(
            self.n_blocks, self.n_landmarks, inp.cut, self.use_variant_prefix, device
        )
        features = self._cell_features(inp, stats, layout)
        x0 = self._embed(features, inp, layout)
        mask = build_attention_mask(layout, inp.valid)
        n_heads = self.blocks[0].attn.num_heads
        attn_mask = (
            mask.unsqueeze(1)
            .expand(-1, n_heads, -1, -1)
            .reshape(-1, mask.shape[1], mask.shape[2])
        )

        injected = self.inject(x0)
        x = x0
        for k in range(self.n_loops):
            block = self.blocks[0] if self.tied else self.blocks[k]
            x = block(x + injected, attn_mask)

        query = x[:, layout.query_index]  # [B, 2P, d_model]
        out = self.readout(self.readout_norm(query))  # [B, 2P, H]
        n_p = self.n_landmarks
        mixer = out[:, :n_p]
        ffn = out[:, n_p:]
        return torch.stack([mixer, ffn], dim=2)  # [B, P, 2, H]


# --------------------------------------------------------------------------------------
# 비교군
# --------------------------------------------------------------------------------------


class PersistencePredictor(nn.Module):
    """V_hat = 0. 학습 parameter가 없습니다."""

    def __init__(self, hidden_size: int, n_landmarks: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_landmarks = n_landmarks
        self.register_buffer("_zero", torch.zeros(1), persistent=False)

    def param_report(self) -> ParamReport:
        return ParamReport(total=0, core=0, input_output=0)

    def forward(self, inp: PredictorInput, stats: NormStats) -> Tensor:  # noqa: ARG002
        batch = inp.batch_size
        return torch.zeros(
            batch, self.n_landmarks, 2, self.hidden_size, device=inp.orig_state.device
        )


class LinearConditionalPredictor(nn.Module):
    """작은 linear conditional baseline.

    landmark별로 [D_norm[cut], U_orig_norm[cut, mixer], U_orig_norm[cut, ffn]]과
    depth one-hot을 받아 V_hat[cut]을 바로 선형 사상합니다. 미래 정보는 쓰지 않습니다.
    """

    def __init__(self, hidden_size: int, n_blocks: int, n_landmarks: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_landmarks = n_landmarks
        self.proj = nn.Linear(3 * hidden_size + n_blocks, 2 * hidden_size)

    def param_report(self) -> ParamReport:
        total = sum(p.numel() for p in self.parameters())
        return ParamReport(total=total, core=0, input_output=total)

    def forward(self, inp: PredictorInput, stats: NormStats) -> Tensor:
        cut = inp.cut
        batch, n_p = inp.batch_size, self.n_landmarks
        device = inp.orig_state.device
        d_cut = inp.var_state_prefix[:, cut] - inp.orig_state[:, cut]  # [B, P, H]
        d_norm = d_cut / max(float(stats.rollout_scale[cut]), stats.floor)
        u_orig = stats.norm_update(inp.orig_updates)[:, cut]  # [B, P, 2, H]
        depth_onehot = torch.zeros(batch, n_p, self.n_blocks, device=device)
        depth_onehot[..., cut] = 1.0
        features = torch.cat([d_norm, u_orig[..., 0, :], u_orig[..., 1, :], depth_onehot], dim=-1)
        out = self.proj(features)  # [B, P, 2H]
        return out.view(batch, n_p, 2, self.hidden_size)


def build_model(cfg, hidden_size: int, n_blocks: int, n_landmarks: int) -> nn.Module:
    """config의 model.name에 맞는 model을 만듭니다.

    loop1 / loop4 / untied4 / m0 는 같은 input/output 계약을 공유하고, persistence와
    linear는 parameter 구조만 다릅니다.
    """
    name = cfg.model.name
    if name == "persistence":
        return PersistencePredictor(hidden_size, n_landmarks)
    if name == "linear":
        return LinearConditionalPredictor(hidden_size, n_blocks, n_landmarks)
    n_loops, tied, use_prefix = cfg.model.n_loops, cfg.model.tied, cfg.model.use_variant_prefix
    if name == "loop1":
        n_loops, tied = 1, True
    elif name == "loop4":
        n_loops, tied = 4, True
    elif name == "untied4":
        n_loops, tied = 4, False
    elif name == "m0":
        use_prefix = False
    return LoopedPredictor(
        hidden_size=hidden_size,
        n_blocks=n_blocks,
        n_landmarks=n_landmarks,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        ffn_dim=cfg.model.ffn_dim,
        dropout=cfg.model.dropout,
        n_loops=n_loops,
        tied=tied,
        use_variant_prefix=use_prefix,
    )
