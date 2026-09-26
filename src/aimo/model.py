"""Looped Transformer core와 Flow view, 그리고 legacy 비교군 model들.

구조는 다음 한 줄로 요약됩니다.

    native vectors
      -> layer 간 공유하는 작은 learned input embedding
      -> shared pre-LN Transformer block 반복 (동일 parameter 객체 4회 호출)
      -> view별 readout (Flow는 native H-space, Behavior는 pair readout)

v2에서는 Behavior view와 Flow view가 **같은 LoopedCore 객체**를 씁니다. core는
embedding, injection, shared block을 모두 담고 있고 두 view는 readout head만 다릅니다.
독립적인 Transformer를 두 개 만들지 않습니다 (`behavior.py` 참고).

LLM의 block index d와 predictor의 loop index k는 서로 다른 개념입니다. loop 하나가
LLM layer 하나를 재현한다고 주장하지 않습니다. 변화 문맥은 단일 z bottleneck이
아니라 sequence hidden states에 유지합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .data import FlowInput, NormStats

# cell 종류. KIND_PAIR는 v2에서 추가된 Behavior pair readout query입니다.
KIND_REF = 0
KIND_OBS = 1
KIND_QUERY = 2
KIND_PAIR = 3
N_KINDS = 4

STREAM_NONE = 0
STREAM_MIXER_TOKEN = 1
STREAM_FFN_TOKEN = 2

# validity를 어느 쪽 Page에서 가져오는지. Behavior는 original/variant validity가 다릅니다.
SIDE_ORIGINAL = 0
SIDE_VARIANT = 1
SIDE_NONE = 2  # query cell: landmark가 없으므로 항상 유효


@dataclass
class ParamReport:
    """total / core / input-output / head별 parameter 수."""

    total: int
    core: int
    input_output: int
    heads: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "core": self.core,
            "input_output": self.input_output,
            "heads": dict(self.heads),
        }


# --------------------------------------------------------------------------------------
# Sequence layout과 attention mask
# --------------------------------------------------------------------------------------


@dataclass
class SequenceLayout:
    """cell 배치와 masking에 필요한 index 정보.

    Flow  cell 순서: [reference depth 0..L] + [observed variant depth 0..cut] + [query 2P].
    Behavior cell 순서: [reference depth 0..L] + [variant depth 0..L] + [pair query 1].
    같은 depth의 landmarks는 하나의 group으로 취급되어 mask row/column 패턴을 공유합니다.
    """

    kind: Tensor  # [S] int64
    depth: Tensor  # [S] int64
    landmark: Tensor  # [S] int64
    stream: Tensor  # [S] int64, flow query cell만 mixer/ffn 구분
    side: Tensor  # [S] int64, validity를 가져올 Page (SIDE_*)
    query_index: Tensor  # [Q] int64, readout을 뽑을 cell 위치
    n_landmarks: int
    cut: int  # query가 읽을 수 있는 최대 variant depth


def build_layout(
    n_blocks: int, n_landmarks: int, cut: int, use_variant_prefix: bool, device: torch.device
) -> SequenceLayout:
    """Flow view layout. query는 cut depth의 mixer/ffn 2P cell입니다."""
    kinds, depths, landmarks, streams, sides = [], [], [], [], []

    def add(kind: int, depth: int, stream: int, side: int) -> None:
        for p in range(n_landmarks):
            kinds.append(kind)
            depths.append(depth)
            landmarks.append(p)
            streams.append(stream)
            sides.append(side)

    for r in range(n_blocks + 1):  # reference: original state 0..L, update r-1
        add(KIND_REF, r, STREAM_NONE, SIDE_ORIGINAL)
    if use_variant_prefix:
        for r in range(cut + 1):  # observed variant: state 0..cut, update r-1
            add(KIND_OBS, r, STREAM_NONE, SIDE_VARIANT)
    query_start = len(kinds)
    add(KIND_QUERY, cut, STREAM_MIXER_TOKEN, SIDE_ORIGINAL)
    add(KIND_QUERY, cut, STREAM_FFN_TOKEN, SIDE_ORIGINAL)
    query_index = torch.arange(query_start, len(kinds), device=device)
    return _finish_layout(
        kinds, depths, landmarks, streams, sides, query_index, n_landmarks, cut, device
    )


def _finish_layout(
    kinds: list[int],
    depths: list[int],
    landmarks: list[int],
    streams: list[int],
    sides: list[int],
    query_index: Tensor,
    n_landmarks: int,
    cut: int,
    device: torch.device,
) -> SequenceLayout:
    def to(xs: list[int]) -> Tensor:
        return torch.tensor(xs, dtype=torch.long, device=device)

    return SequenceLayout(
        kind=to(kinds),
        depth=to(depths),
        landmark=to(landmarks),
        stream=to(streams),
        side=to(sides),
        query_index=query_index,
        n_landmarks=n_landmarks,
        cut=cut,
    )


def cell_validity(layout: SequenceLayout, orig_valid: Tensor, var_valid: Tensor) -> Tensor:
    """[B, S] cell validity. original/variant validity를 각각의 side에서 가져옵니다.

    query cell은 landmark 관측이 아니므로 항상 유효합니다.
    """
    per_landmark = torch.stack(
        [orig_valid, var_valid, torch.ones_like(orig_valid)], dim=0
    )  # [3, B, P]
    return per_landmark[layout.side, :, layout.landmark].transpose(0, 1).contiguous()


def build_attention_mask(layout: SequenceLayout, cell_valid: Tensor) -> Tensor:
    """attend 가능 여부 mask. 반환값은 True가 '차단'인 bool [B, S, S]입니다.

    규칙 (Flow와 Behavior가 동일하며 모든 loop에서 같습니다):
      reference  -> reference만 읽습니다 (variant 관측을 전혀 읽지 않습니다).
      observed r -> reference 전체와 observed depth <= r.
      query      -> reference 전체와 observed depth <= cut.
      reference/observed는 어떤 query cell도 읽지 않습니다.
    Behavior는 cut = L로 두므로 pair query가 variant 전체 depth를 읽습니다.
    padding cell은 key에서 제외하고, 전부 차단된 row는 self를 열어 NaN을 막습니다.
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
    # KIND_QUERY와 KIND_PAIR 모두 query row 규칙을 씁니다.
    allowed = torch.where(kind_i >= KIND_QUERY, allow_query_row, allowed)  # [S, S]

    batch = cell_valid.shape[0]
    if cell_valid.shape[1] != n_seq:
        raise ValueError(f"cell_valid must be [B, {n_seq}], got {tuple(cell_valid.shape)}")
    allowed = allowed.unsqueeze(0).expand(batch, n_seq, n_seq).clone()
    allowed &= cell_valid.unsqueeze(1)

    empty_rows = ~allowed.any(dim=-1)
    if bool(empty_rows.any()):
        eye = torch.eye(n_seq, dtype=torch.bool, device=allowed.device).unsqueeze(0)
        allowed |= empty_rows.unsqueeze(-1) & eye
    return ~allowed


def expand_head_mask(mask: Tensor, n_heads: int) -> Tensor:
    """[B, S, S] bool mask를 nn.MultiheadAttention의 [B*heads, S, S]로 펼칩니다."""
    return (
        mask.unsqueeze(1).expand(-1, n_heads, -1, -1).reshape(-1, mask.shape[1], mask.shape[2])
    )


# --------------------------------------------------------------------------------------
# Shared pre-LN block과 shared core
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


class LoopedCore(nn.Module):
    """공유 input embedding + shared block loop. Behavior와 Flow가 같은 객체를 씁니다.

        x_0     = emb_norm(embed(observations))
        x_{k+1} = Block(x_k + inject(x_0))          k = 0 .. n_loops-1

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
    ) -> None:
        super().__init__()
        if n_loops < 1:
            raise ValueError("n_loops must be >= 1")
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_landmarks = n_landmarks
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_loops = n_loops
        self.tied = tied

        # cell feature = [state, update_mixer, update_ffn] 의 native vector 연결.
        self.value_proj = nn.Linear(3 * hidden_size, d_model)
        self.kind_emb = nn.Embedding(N_KINDS, d_model)
        self.stream_emb = nn.Embedding(3, d_model)
        self.depth_emb = nn.Embedding(n_blocks + 1, d_model)
        self.landmark_emb = nn.Embedding(n_landmarks, d_model)
        # relative position과 query까지의 상대 depth는 연속값으로 넣습니다.
        self.scalar_proj = nn.Linear(3, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.inject = nn.Linear(d_model, d_model, bias=False)

        if tied:
            block = SharedBlock(d_model, n_heads, ffn_dim, dropout)
            self.blocks = nn.ModuleList([block])  # 동일 parameter 객체를 n_loops회 호출
        else:
            self.blocks = nn.ModuleList(
                [SharedBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_loops)]
            )

    def embedding_parameters(self) -> list[nn.Module]:
        return [
            self.value_proj,
            self.kind_emb,
            self.stream_emb,
            self.depth_emb,
            self.landmark_emb,
            self.scalar_proj,
            self.emb_norm,
            self.inject,
        ]

    def embed(self, features: Tensor, layout: SequenceLayout, rel_pos: Tensor) -> Tensor:
        """[B, S, 3H] feature와 [B, S] relative position에서 x_0를 만듭니다."""
        batch = features.shape[0]
        depth = layout.depth
        rel_cut = (depth.float() - float(layout.cut)) / max(self.n_blocks, 1)
        at_cut = (depth == layout.cut).float()
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

    def run_loops(self, x0: Tensor, mask: Tensor) -> Tensor:
        """x_0에서 n_loops회 block을 돌립니다. 매 loop에 x_0를 다시 주입합니다."""
        attn_mask = expand_head_mask(mask, self.n_heads)
        injected = self.inject(x0)
        x = x0
        for k in range(self.n_loops):
            block = self.blocks[0] if self.tied else self.blocks[k]
            x = block(x + injected, attn_mask)
        return x


def stacked_cell_features(
    state: Tensor, updates: Tensor, stats: NormStats, hidden: int
) -> Tensor:
    """[B, D, P, H] state와 [B, D-1, P, 2, H] updates에서 [B, D, P, 3H] cell feature.

    indexing 규칙: depth r cell에는 state[r]과 '이미 완료된' update[r-1]만 넣습니다.
    r = 0의 update 자리는 0입니다. 따라서 예측 target U[d]가 depth d cell에 들어가지
    않습니다.
    """
    batch, depth, n_p = state.shape[0], state.shape[1], state.shape[2]
    normed_state = stats.norm_state(state)
    zeros = torch.zeros(batch, 1, n_p, 2, hidden, device=state.device)
    if updates.shape[1] > 0:
        prev = torch.cat([zeros, stats.norm_update(updates)], dim=1)
    else:
        prev = zeros
    prev = prev[:, :depth]
    return torch.cat([normed_state, prev[..., 0, :], prev[..., 1, :]], dim=-1)


# --------------------------------------------------------------------------------------
# Flow view
# --------------------------------------------------------------------------------------


class LoopedPredictor(nn.Module):
    """Flow view: cut d에서 V_hat[d] = [P, 2, H]를 normalized space로 예측합니다.

    core와 readout은 모든 depth가 공유하는 단일 module입니다. readout은 native
    H-space로 직접 사상하며 z bottleneck이나 random projection을 쓰지 않습니다.
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
        core: LoopedCore | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_landmarks = n_landmarks
        self.use_variant_prefix = use_variant_prefix
        self.core = core or LoopedCore(
            hidden_size=hidden_size,
            n_blocks=n_blocks,
            n_landmarks=n_landmarks,
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            n_loops=n_loops,
            tied=tied,
        )
        # native-space flow readout (layer 간 공유)
        self.readout_norm = nn.LayerNorm(self.core.d_model)
        self.readout = nn.Linear(self.core.d_model, hidden_size)

    # 하위 호환: 기존 코드가 model.n_loops / model.tied / model.blocks를 봅니다.
    @property
    def n_loops(self) -> int:
        return self.core.n_loops

    @property
    def tied(self) -> bool:
        return self.core.tied

    @property
    def blocks(self) -> nn.ModuleList:
        return self.core.blocks

    def param_report(self) -> ParamReport:
        core = sum(p.numel() for p in self.core.blocks.parameters())
        io = sum(p.numel() for m in self.core.embedding_parameters() for p in m.parameters())
        flow_head = sum(p.numel() for p in self.readout_norm.parameters()) + sum(
            p.numel() for p in self.readout.parameters()
        )
        total = sum(p.numel() for p in self.parameters())
        return ParamReport(
            total=total, core=core, input_output=io, heads={"flow": flow_head}
        )

    def _cell_features(
        self, inp: FlowInput, stats: NormStats, layout: SequenceLayout
    ) -> Tensor:
        """[B, S, 3H] cell feature. observed cell은 var_updates_prefix[0:cut]만 소비합니다."""
        batch = inp.batch_size
        hidden = self.hidden_size
        n_p = self.n_landmarks
        device = inp.orig_state.device

        pieces = [stacked_cell_features(inp.orig_state, inp.orig_updates, stats, hidden)]
        if self.use_variant_prefix:
            pieces.append(
                stacked_cell_features(
                    inp.var_state_prefix, inp.var_updates_prefix, stats, hidden
                )
            )
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

    def forward(self, inp: FlowInput, stats: NormStats) -> Tensor:
        """normalized V_hat[cut], shape [B, P, 2, H]."""
        if inp.n_blocks != self.n_blocks:
            raise ValueError(f"expected {self.n_blocks} blocks, got {inp.n_blocks}")
        device = inp.orig_state.device
        layout = build_layout(
            self.n_blocks, self.n_landmarks, inp.cut, self.use_variant_prefix, device
        )
        features = self._cell_features(inp, stats, layout)
        rel_pos = inp.relative_positions[:, layout.landmark]  # [B, S]
        x0 = self.core.embed(features, layout, rel_pos)
        # Flow는 original/variant 공통 valid landmark만 씁니다.
        mask = build_attention_mask(layout, inp.valid[:, layout.landmark])
        x = self.core.run_loops(x0, mask)

        query = x[:, layout.query_index]  # [B, 2P, d_model]
        out = self.readout(self.readout_norm(query))  # [B, 2P, H]
        n_p = self.n_landmarks
        return torch.stack([out[:, :n_p], out[:, n_p:]], dim=2)  # [B, P, 2, H]


# --------------------------------------------------------------------------------------
# 비교군 (flow legacy)
# --------------------------------------------------------------------------------------


class PersistencePredictor(nn.Module):
    """V_hat = 0. 학습 parameter가 없습니다."""

    def __init__(self, hidden_size: int, n_landmarks: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_landmarks = n_landmarks
        self.register_buffer("_zero", torch.zeros(1), persistent=False)

    def param_report(self) -> ParamReport:
        return ParamReport(total=0, core=0, input_output=0, heads={})

    def forward(self, inp: FlowInput, stats: NormStats) -> Tensor:  # noqa: ARG002
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
        return ParamReport(total=total, core=0, input_output=total, heads={})

    def forward(self, inp: FlowInput, stats: NormStats) -> Tensor:
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


FLOW_MODEL_NAMES = ("persistence", "linear", "m0", "loop1", "loop4", "untied4")


def build_model(cfg, hidden_size: int, n_blocks: int, n_landmarks: int) -> nn.Module:
    """config의 model.name에 맞는 model을 만듭니다.

    flow-only 이름(loop1 / loop4 / untied4 / m0 / persistence / linear)은 LoopedPredictor
    계열을 돌려주고, behavior가 필요한 이름은 behavior.AimoModel을 돌려줍니다.
    """
    name = cfg.model.name
    if name not in FLOW_MODEL_NAMES:
        from .behavior import build_behavior_model  # 지역 import로 순환 참조를 피합니다.

        return build_behavior_model(cfg, hidden_size, n_blocks, n_landmarks)
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
