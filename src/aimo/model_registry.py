"""제출과 추출이 공유하는 model registry.

layout은 decoder layer에서 actual residual update를 읽는 방식입니다. 확인된 module 이름만
허용하고, 알 수 없는 `model_type`은 추측하지 않고 `SCHEMA_INCOMPATIBLE`로 처리합니다.

- `pre_norm`: `h_mid = h + Mixer(norm(h))`, `h_out = h_mid + FFN(norm(h_mid))`.
  Qwen2 / Qwen3 / Qwen3.5 hybrid(linear-attention + full-attention Mixer)가 여기에 속합니다.
  `h_mid`는 `post_attention_layernorm`의 입력입니다.
- `post_norm`: `h_mid = h + norm(Mixer(h))`, `h_out = h_mid + norm(FFN(h_mid))`.
  OLMo-2 계열 구조이며 `h_mid`는 `mlp`의 입력입니다.

actual update는 residual 차이(`h_mid - h_in`, `h_out - h_mid`)로 정의하고, module hook
출력은 그 값과 일치하는지 교차 검증하는 audit 값으로만 씁니다. 일반 attention output hook을
residual update로 그대로 쓰지 않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .official_contract import MAIN_ONLY_MODELS, SMALL_TRACK_MODELS

LAYOUT_PRE_NORM = "pre_norm"
LAYOUT_POST_NORM = "post_norm"

ADAPTER_IMPLEMENTED_UNVERIFIED = "ADAPTER_IMPLEMENTED / REAL_WEIGHTS_UNVERIFIED"
ADAPTER_NONE = "NO_ADAPTER"


@dataclass(frozen=True)
class LayoutSpec:
    """actual residual update를 읽기 위한 module 위치."""

    name: str
    mid_module: str  # 이 module의 입력이 h_mid (Mixer update가 더해진 residual)
    mixer_update_modules: tuple[str, ...]  # 출력이 Mixer update여야 하는 module (audit)
    ffn_update_module: str  # 출력이 FFN update여야 하는 module (audit)


LAYOUTS = {
    LAYOUT_PRE_NORM: LayoutSpec(
        name=LAYOUT_PRE_NORM,
        mid_module="post_attention_layernorm",
        mixer_update_modules=("linear_attn", "self_attn"),
        ffn_update_module="mlp",
    ),
    LAYOUT_POST_NORM: LayoutSpec(
        name=LAYOUT_POST_NORM,
        mid_module="mlp",
        mixer_update_modules=("post_attention_layernorm",),
        ffn_update_module="post_feedforward_layernorm",
    ),
}

# Mixer 종류는 layer의 module 이름으로 기록합니다. input feature로 쓰지 않습니다.
MIXER_KIND_BY_ATTR = {"linear_attn": "linear_attention", "self_attn": "full_attention"}

# config.model_type -> layout. 확인한 계열만 둡니다.
MODEL_TYPE_LAYOUT = {
    "qwen2": LAYOUT_PRE_NORM,
    "qwen3": LAYOUT_PRE_NORM,
    "qwen3_next": LAYOUT_PRE_NORM,
    "qwen3_5": LAYOUT_PRE_NORM,
    "qwen3_5_text": LAYOUT_PRE_NORM,
    "llama": LAYOUT_PRE_NORM,
    "olmo2": LAYOUT_POST_NORM,
    "olmo3": LAYOUT_POST_NORM,
    # CPU 검증 전용 toy model.
    "aimo_toy_pre_norm": LAYOUT_PRE_NORM,
    "aimo_toy_post_norm": LAYOUT_POST_NORM,
}


@dataclass(frozen=True)
class ModelEntry:
    """공식 track model 하나. revision은 artifact가 실제 load한 commit으로 pin합니다."""

    model_id: str
    family: str
    layout: str | None
    adapter_status: str
    small_track: bool
    primary: bool = False
    # effort별 chat template kwargs. 확인되지 않은 effort 조절은 넣지 않고 metadata로만 씁니다.
    template_kwargs: tuple[tuple[str, object], ...] = ()
    effort_policy: str = "metadata_only"


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "Qwen/Qwen3.5-4B": ModelEntry(
        model_id="Qwen/Qwen3.5-4B",
        family="qwen3_5",
        layout=LAYOUT_PRE_NORM,
        adapter_status=ADAPTER_IMPLEMENTED_UNVERIFIED,
        small_track=True,
        primary=True,
    ),
    "Skywork/Skywork-OR1-Math-7B": ModelEntry(
        model_id="Skywork/Skywork-OR1-Math-7B",
        family="qwen2",
        layout=LAYOUT_PRE_NORM,
        adapter_status=ADAPTER_IMPLEMENTED_UNVERIFIED,
        small_track=True,
    ),
    "allenai/Olmo-3-7B-Think": ModelEntry(
        model_id="allenai/Olmo-3-7B-Think",
        family="olmo3",
        layout=LAYOUT_POST_NORM,
        adapter_status=ADAPTER_IMPLEMENTED_UNVERIFIED,
        small_track=True,
    ),
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": ModelEntry(
        model_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        family="qwen3",
        layout=LAYOUT_PRE_NORM,
        adapter_status=ADAPTER_IMPLEMENTED_UNVERIFIED,
        small_track=True,
    ),
    "openai/gpt-oss-120b": ModelEntry(
        model_id="openai/gpt-oss-120b",
        family="gpt_oss",
        layout=None,
        adapter_status=ADAPTER_NONE,
        small_track=False,
    ),
}

assert tuple(k for k, v in MODEL_REGISTRY.items() if v.small_track) == SMALL_TRACK_MODELS
assert tuple(k for k, v in MODEL_REGISTRY.items() if not v.small_track) == MAIN_ONLY_MODELS


def layout_for_model_type(model_type: str) -> LayoutSpec:
    from .native_page import SchemaIncompatible

    name = MODEL_TYPE_LAYOUT.get(model_type)
    if name is None:
        raise SchemaIncompatible(
            f"SCHEMA_INCOMPATIBLE: model_type {model_type!r} has no confirmed residual layout; "
            "add an adapter only after checking the decoder layer source"
        )
    return LAYOUTS[name]
