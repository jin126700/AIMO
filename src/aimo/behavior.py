"""Behavior view: full-page pair readout, signed drop head, panel set pooling.

Behavior와 Flow는 **같은 LoopedCore 객체**를 공유하고 readout head만 다릅니다. 두
view는 서로 다른 forward이며, Behavior의 hidden state나 KV cache를 Flow에 재사용하지
않습니다. 그래서 full-page behavior 경로를 통해 variant future가 Flow로 새지 않습니다.

Behavior sequence (cut = L로 두어 pair query가 variant 전체 depth를 읽습니다):

    [original reference depth 0..L] [variant depth 0..L] [pair readout query 1]

Mask 규칙은 `model.build_attention_mask`와 동일합니다.
  - original reference는 original만 읽습니다.
  - variant depth r은 original 전체와 variant depth <= r을 읽습니다.
  - pair readout query는 original/variant 전체 관측을 읽습니다.
  - reference/variant row는 readout query를 읽지 않습니다.
  - 모든 loop에서 같은 mask를 씁니다.

학습된 pair score나 pooling weight를 개별 변형의 causal importance라고 해석하지
않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .data import BehaviorInput, NormStats
from .model import (
    KIND_OBS,
    KIND_PAIR,
    KIND_REF,
    SIDE_NONE,
    SIDE_ORIGINAL,
    SIDE_VARIANT,
    STREAM_NONE,
    LoopedCore,
    LoopedPredictor,
    ParamReport,
    SequenceLayout,
    _finish_layout,
    build_attention_mask,
    cell_validity,
    stacked_cell_features,
)

# head 학습 여부 flag의 순서. checkpoint에 함께 저장합니다.
HEAD_NAMES = ("pair_drop", "robust", "max_drop")


def build_behavior_layout(
    n_blocks: int, n_landmarks: int, device: torch.device
) -> SequenceLayout:
    """Behavior view layout. query는 pair readout cell 1개입니다."""
    kinds, depths, landmarks, streams, sides = [], [], [], [], []

    def add_depth(kind: int, depth: int, side: int) -> None:
        for p in range(n_landmarks):
            kinds.append(kind)
            depths.append(depth)
            landmarks.append(p)
            streams.append(STREAM_NONE)
            sides.append(side)

    for r in range(n_blocks + 1):
        add_depth(KIND_REF, r, SIDE_ORIGINAL)
    for r in range(n_blocks + 1):  # variant 전체 depth: state[0:L+1], updates[0:L]
        add_depth(KIND_OBS, r, SIDE_VARIANT)
    query_start = len(kinds)
    kinds.append(KIND_PAIR)
    depths.append(n_blocks)
    landmarks.append(0)  # pair query는 특정 landmark의 관측이 아닙니다.
    streams.append(STREAM_NONE)
    sides.append(SIDE_NONE)
    query_index = torch.arange(query_start, len(kinds), device=device)
    return _finish_layout(
        kinds, depths, landmarks, streams, sides, query_index, n_landmarks, n_blocks, device
    )


@dataclass
class BehaviorOutput:
    """pair 단위 출력. Z는 R^d_model의 pair representation입니다."""

    z: Tensor  # [N, d_model]
    pair_drop: Tensor  # [N] float, tanh로 [-1, 1]


@dataclass
class PanelOutput:
    """panel 단위 출력. 구성원이 없는 panel은 NaN이고 panel_valid가 False입니다."""

    robust_logit: Tensor  # [B]
    robust_prob: Tensor  # [B]
    max_drop: Tensor  # [B], max(0, max_j d_hat_ij)
    pooling_weights: Tensor  # [B, M]
    pair_drop: Tensor  # [B, M], padding 자리는 NaN
    panel_valid: Tensor  # [B] bool


class BehaviorHeads(nn.Module):
    """pair drop head와 original-panel robustness pooling.

    pooling은 작은 learned attention pooling이며 variant 순서 embedding을 쓰지 않습니다.
    따라서 같은 panel의 variants 순서를 바꿔도 결과가 같습니다 (permutation invariance).
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.pair_norm = nn.LayerNorm(d_model)
        self.drop_head = nn.Linear(d_model, 1)
        self.pool_score = nn.Linear(d_model, 1)
        self.pool_value = nn.Linear(d_model, d_model)
        self.pool_norm = nn.LayerNorm(d_model)
        self.robust_head = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def pair_embedding(self, query: Tensor) -> Tensor:
        """[N, 1, d_model] readout cell -> [N, d_model] Z_ij."""
        return self.pair_norm(query.squeeze(1))

    def pair_drop(self, z: Tensor) -> Tensor:
        """signed drop d_hat_ij in [-1, 1]."""
        return torch.tanh(self.drop_head(self.dropout(z))).squeeze(-1)

    def panel(self, z_panel: Tensor, panel_mask: Tensor, pair_drop: Tensor) -> PanelOutput:
        """[B, M, d_model] Z와 [B, M] mask에서 panel 출력을 만듭니다."""
        scores = self.pool_score(z_panel).squeeze(-1)  # [B, M]
        # padding slot은 아주 낮은 점수로 눌러 softmax에서 제외합니다 (-inf는 NaN 위험).
        scores = scores.masked_fill(~panel_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        weights = weights * panel_mask  # 빈 panel은 전부 0이 됩니다.
        pooled = (weights.unsqueeze(-1) * self.pool_value(z_panel)).sum(dim=1)
        logit = self.robust_head(self.pool_norm(pooled)).squeeze(-1)
        panel_valid = panel_mask.any(dim=1)
        nan = torch.full_like(logit, float("nan"))
        logit = torch.where(panel_valid, logit, nan)
        prob = torch.sigmoid(logit)
        masked_drop = torch.where(
            panel_mask, pair_drop, torch.full_like(pair_drop, float("-inf"))
        )
        max_drop = masked_drop.max(dim=1).values.clamp_min(0.0)
        max_drop = torch.where(panel_valid, max_drop, nan)
        return PanelOutput(
            robust_logit=logit,
            robust_prob=prob,
            max_drop=max_drop,
            pooling_weights=weights,
            pair_drop=torch.where(
                panel_mask, pair_drop, torch.full_like(pair_drop, float("nan"))
            ),
            panel_valid=panel_valid,
        )


class AimoModel(nn.Module):
    """Behavior + Flow joint model. 두 view가 같은 LoopedCore 객체를 공유합니다.

    `task`는 어떤 view를 학습하는지만 나타내며 parameter 구조는 같습니다.
      behavior : behavior head만 학습 (flow loss 없음)
      joint    : L = L_behavior + 0.1 * L_flow  (primary)
      flow     : flow only (legacy/auxiliary 비교)
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
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_landmarks = n_landmarks
        self.flow = LoopedPredictor(
            hidden_size=hidden_size,
            n_blocks=n_blocks,
            n_landmarks=n_landmarks,
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            n_loops=n_loops,
            tied=tied,
            use_variant_prefix=use_variant_prefix,
        )
        self.behavior = BehaviorHeads(d_model=self.flow.core.d_model, dropout=dropout)
        # 실제 label로 학습된 head만 1이 됩니다. 무작위 초기 head의 출력을 검증된
        # robust probability처럼 내보내지 않기 위한 표시입니다.
        self.register_buffer("head_trained", torch.zeros(len(HEAD_NAMES)))

    # -- 공유 core -------------------------------------------------------------------
    @property
    def core(self) -> LoopedCore:
        return self.flow.core

    def mark_trained(self, name: str) -> None:
        self.head_trained[HEAD_NAMES.index(name)] = 1.0

    def is_trained(self, name: str) -> bool:
        return bool(self.head_trained[HEAD_NAMES.index(name)] > 0)

    def trained_heads(self) -> dict[str, bool]:
        return {name: self.is_trained(name) for name in HEAD_NAMES}

    def param_report(self) -> ParamReport:
        flow_report = self.flow.param_report()
        behavior_head = sum(p.numel() for p in self.behavior.parameters())
        heads = dict(flow_report.heads)
        heads["behavior"] = behavior_head
        return ParamReport(
            total=sum(p.numel() for p in self.parameters()),
            core=flow_report.core,
            input_output=flow_report.input_output,
            heads=heads,
        )

    # -- Flow view -------------------------------------------------------------------
    def forward_flow(self, inp, stats: NormStats) -> Tensor:
        return self.flow(inp, stats)

    def forward(self, inp, stats: NormStats) -> Tensor:
        """기존 flow API 호환: flow view를 그대로 호출합니다."""
        return self.flow(inp, stats)

    # -- Behavior view ---------------------------------------------------------------
    def _behavior_features(
        self, inp: BehaviorInput, stats: NormStats, layout: SequenceLayout
    ) -> Tensor:
        """[N, S, 3H] cell feature. variant는 state[0:L+1], updates[0:L] 전체를 씁니다."""
        hidden = self.hidden_size
        batch = inp.batch_size
        device = inp.orig_state.device
        ref = stacked_cell_features(inp.orig_state, inp.orig_updates, stats, hidden)
        var = stacked_cell_features(inp.var_state, inp.var_updates, stats, hidden)
        query = torch.zeros(batch, 1, 3 * hidden, device=device)
        features = torch.cat(
            [ref.reshape(batch, -1, 3 * hidden), var.reshape(batch, -1, 3 * hidden), query], dim=1
        )
        if features.shape[1] != layout.kind.shape[0]:
            raise RuntimeError(
                f"behavior sequence length mismatch: features {features.shape[1]} "
                f"vs layout {layout.kind.shape[0]}"
            )
        cell_valid = cell_validity(layout, inp.orig_valid, inp.var_valid)
        return features * cell_valid.unsqueeze(-1)

    def forward_behavior(self, inp: BehaviorInput, stats: NormStats) -> BehaviorOutput:
        """pair 단위 Z와 signed drop을 계산합니다."""
        if inp.n_blocks != self.n_blocks:
            raise ValueError(f"expected {self.n_blocks} blocks, got {inp.n_blocks}")
        device = inp.orig_state.device
        layout = build_behavior_layout(self.n_blocks, self.n_landmarks, device)
        features = self._behavior_features(inp, stats, layout)
        rel_pos = _behavior_rel_positions(inp, layout)
        x0 = self.core.embed(features, layout, rel_pos)
        mask = build_attention_mask(layout, cell_validity(layout, inp.orig_valid, inp.var_valid))
        x = self.core.run_loops(x0, mask)
        z = self.behavior.pair_embedding(x[:, layout.query_index])
        return BehaviorOutput(z=z, pair_drop=self.behavior.pair_drop(z))

    def panel_outputs(
        self,
        out: BehaviorOutput,
        pair_panel: Tensor,
        pair_slot: Tensor,
        panel_mask: Tensor,
    ) -> PanelOutput:
        """pair 출력을 [B, M] panel 좌표로 흩뿌린 뒤 set pooling을 적용합니다."""
        n_panels, max_members = panel_mask.shape
        z_panel = out.z.new_zeros(n_panels, max_members, out.z.shape[-1])
        drops = out.pair_drop.new_zeros(n_panels, max_members)
        z_panel[pair_panel, pair_slot] = out.z
        drops[pair_panel, pair_slot] = out.pair_drop
        return self.behavior.panel(z_panel, panel_mask, drops)


def _behavior_rel_positions(inp: BehaviorInput, layout: SequenceLayout) -> Tensor:
    """[N, S] relative position. side에 맞는 Page의 위치를 쓰고 query는 1.0입니다."""
    ones = torch.ones_like(inp.orig_relative_positions)
    stacked = torch.stack(
        [inp.orig_relative_positions, inp.var_relative_positions, ones], dim=0
    )  # [3, N, P]
    return stacked[layout.side, :, layout.landmark].transpose(0, 1).contiguous()


def build_behavior_model(cfg, hidden_size: int, n_blocks: int, n_landmarks: int) -> nn.Module:
    """config의 model.name에서 behavior/joint model 또는 behavior 비교군을 만듭니다."""
    name = cfg.model.name
    if name in BEHAVIOR_BASELINES:
        return BEHAVIOR_BASELINES[name](hidden_size, n_blocks, n_landmarks)
    n_loops, tied, use_prefix = cfg.model.n_loops, cfg.model.tied, cfg.model.use_variant_prefix
    if name in ("behavior_loop1", "joint_loop1"):
        n_loops, tied = 1, True
    elif name in ("behavior", "joint", "behavior_loop4", "joint_loop4"):
        n_loops, tied = 4, True
    elif name in ("behavior_untied4", "joint_untied4"):
        n_loops, tied = 4, False
    elif name in ("behavior_m0", "joint_m0"):
        # M0는 flow에서 variant prefix를 받지 않습니다. behavior에서도 variant 관측을
        # 읽지 않도록 아래 M0BehaviorModel이 variant cell을 original로 대체합니다.
        use_prefix = False
    else:
        raise ValueError(f"unknown behavior model name {name!r}")
    cls = M0BehaviorModel if name in ("behavior_m0", "joint_m0") else AimoModel
    return cls(
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


class M0BehaviorModel(AimoModel):
    """original-only 비교군.

    variant 관측을 original 관측으로 바꿔 넣으므로 variant state/길이/validity/ID/count가
    전혀 들어가지 않습니다. sequence 길이도 variant 수와 무관하게 고정입니다.
    """

    def forward_behavior(self, inp: BehaviorInput, stats: NormStats) -> BehaviorOutput:
        masked = BehaviorInput(
            orig_state=inp.orig_state,
            orig_updates=inp.orig_updates,
            var_state=inp.orig_state,
            var_updates=inp.orig_updates,
            orig_valid=inp.orig_valid,
            var_valid=inp.orig_valid,
            orig_relative_positions=inp.orig_relative_positions,
            var_relative_positions=inp.orig_relative_positions,
            orig_token_offsets=inp.orig_token_offsets,
            var_token_offsets=inp.orig_token_offsets,
        )
        return super().forward_behavior(masked, stats)


# --------------------------------------------------------------------------------------
# Behavior 비교군 (core 없음)
# --------------------------------------------------------------------------------------


class _BaseBehaviorBaseline(nn.Module):
    """core를 쓰지 않는 behavior 비교군의 공통 부분.

    flow view는 지원하지 않으므로 task=behavior에서만 씁니다.
    """

    def __init__(self, hidden_size: int, n_blocks: int, n_landmarks: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_landmarks = n_landmarks
        self.register_buffer("head_trained", torch.zeros(len(HEAD_NAMES)))

    def mark_trained(self, name: str) -> None:
        self.head_trained[HEAD_NAMES.index(name)] = 1.0

    def is_trained(self, name: str) -> bool:
        return bool(self.head_trained[HEAD_NAMES.index(name)] > 0)

    def trained_heads(self) -> dict[str, bool]:
        return {name: self.is_trained(name) for name in HEAD_NAMES}

    def forward_flow(self, inp, stats: NormStats) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__}는 behavior 비교군이므로 flow view를 지원하지 않습니다"
        )

    def param_report(self) -> ParamReport:
        total = sum(p.numel() for p in self.parameters())
        return ParamReport(total=total, core=0, input_output=0, heads={"behavior": total})

    def panel_outputs(
        self, out: BehaviorOutput, pair_panel: Tensor, pair_slot: Tensor, panel_mask: Tensor
    ) -> PanelOutput:
        raise NotImplementedError


class ConstantBehaviorModel(_BaseBehaviorBaseline):
    """constant/prior baseline. 입력을 보지 않고 상수 drop과 상수 robust logit만 학습합니다."""

    def __init__(self, hidden_size: int, n_blocks: int, n_landmarks: int) -> None:
        super().__init__(hidden_size, n_blocks, n_landmarks)
        self.drop_bias = nn.Parameter(torch.zeros(1))
        self.robust_bias = nn.Parameter(torch.zeros(1))

    def forward_behavior(self, inp: BehaviorInput, stats: NormStats) -> BehaviorOutput:  # noqa: ARG002
        n = inp.batch_size
        drop = torch.tanh(self.drop_bias).expand(n)
        return BehaviorOutput(z=drop.unsqueeze(-1), pair_drop=drop)

    def panel_outputs(
        self, out: BehaviorOutput, pair_panel: Tensor, pair_slot: Tensor, panel_mask: Tensor
    ) -> PanelOutput:
        n_panels, max_members = panel_mask.shape
        drops = out.pair_drop.new_zeros(n_panels, max_members)
        drops[pair_panel, pair_slot] = out.pair_drop
        panel_valid = panel_mask.any(dim=1)
        logit = self.robust_bias.expand(n_panels).clone()
        nan = torch.full_like(logit, float("nan"))
        logit = torch.where(panel_valid, logit, nan)
        masked = torch.where(panel_mask, drops, torch.full_like(drops, float("-inf")))
        max_drop = torch.where(panel_valid, masked.max(dim=1).values.clamp_min(0.0), nan)
        return PanelOutput(
            robust_logit=logit,
            robust_prob=torch.sigmoid(logit),
            max_drop=max_drop,
            pooling_weights=panel_mask.to(drops.dtype)
            / panel_mask.sum(dim=1, keepdim=True).clamp_min(1),
            pair_drop=torch.where(panel_mask, drops, torch.full_like(drops, float("nan"))),
            panel_valid=panel_valid,
        )


class RawChangeBehaviorModel(_BaseBehaviorBaseline):
    """작은 raw-change baseline.

    pair마다 세 개의 scalar만 봅니다: 정규화된 최종 state 차이 크기, 정규화된 update 차이
    누적 크기, 공통 valid landmark 비율. Looped core도 attention pooling도 쓰지 않습니다.
    """

    N_FEATURES = 3

    def __init__(self, hidden_size: int, n_blocks: int, n_landmarks: int) -> None:
        super().__init__(hidden_size, n_blocks, n_landmarks)
        self.drop_head = nn.Linear(self.N_FEATURES, 1)
        self.robust_head = nn.Linear(self.N_FEATURES, 1)

    def _features(self, inp: BehaviorInput, stats: NormStats) -> Tensor:
        valid = (inp.orig_valid & inp.var_valid).float()  # [N, P]
        denom = valid.sum(dim=1).clamp_min(1.0)
        state_diff = inp.var_state[:, -1] - inp.orig_state[:, -1]  # [N, P, H]
        state_scale = max(float(stats.rollout_scale[-1]), stats.floor)
        state_norm = (state_diff.norm(dim=-1) * valid).sum(dim=1) / denom / state_scale
        update_diff = (inp.var_updates - inp.orig_updates).sum(dim=(1, 3))  # [N, P, H]
        update_scale = max(float(stats.target_scale.clamp_min(stats.floor).mean()), stats.floor)
        update_norm = (update_diff.norm(dim=-1) * valid).sum(dim=1) / denom / update_scale
        frac = denom / max(self.n_landmarks, 1)
        return torch.stack([state_norm, update_norm, frac], dim=-1)

    def forward_behavior(self, inp: BehaviorInput, stats: NormStats) -> BehaviorOutput:
        features = self._features(inp, stats)
        return BehaviorOutput(
            z=features, pair_drop=torch.tanh(self.drop_head(features)).squeeze(-1)
        )

    def panel_outputs(
        self, out: BehaviorOutput, pair_panel: Tensor, pair_slot: Tensor, panel_mask: Tensor
    ) -> PanelOutput:
        n_panels, max_members = panel_mask.shape
        feats = out.z.new_zeros(n_panels, max_members, out.z.shape[-1])
        drops = out.pair_drop.new_zeros(n_panels, max_members)
        feats[pair_panel, pair_slot] = out.z
        drops[pair_panel, pair_slot] = out.pair_drop
        weights = panel_mask.to(feats.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (weights.unsqueeze(-1) * feats).sum(dim=1)  # 순서와 무관한 평균 pooling
        logit = self.robust_head(pooled).squeeze(-1)
        panel_valid = panel_mask.any(dim=1)
        nan = torch.full_like(logit, float("nan"))
        logit = torch.where(panel_valid, logit, nan)
        masked = torch.where(panel_mask, drops, torch.full_like(drops, float("-inf")))
        max_drop = torch.where(panel_valid, masked.max(dim=1).values.clamp_min(0.0), nan)
        return PanelOutput(
            robust_logit=logit,
            robust_prob=torch.sigmoid(logit),
            max_drop=max_drop,
            pooling_weights=weights,
            pair_drop=torch.where(panel_mask, drops, torch.full_like(drops, float("nan"))),
            panel_valid=panel_valid,
        )


BEHAVIOR_BASELINES = {"constant": ConstantBehaviorModel, "raw_change": RawChangeBehaviorModel}
