"""Qwen decoder에서 Page를 뽑는 adapter.

Qwen 계열 decoder layer의 forward는 다음 순서입니다.

    residual = h
    h = residual + self_attn(input_layernorm(h))        # stream 0 = Mixer
    residual = h
    h = residual + mlp(post_attention_layernorm(h))     # stream 1 = FFN

따라서 Page 계약은 hook 세 곳으로 정확히 채워집니다.

    state[d]      = layer d의 input hidden state
    updates[d, 0] = layer d의 self_attn 출력
    updates[d, 1] = layer d의 mlp 출력
    state[L]      = 마지막 layer의 출력 (final LayerNorm 이전)

state[L]은 final LayerNorm 이후의 hidden state가 아닙니다. 둘을 혼동하지 않습니다.

실제 Qwen3-4B weights는 로컬에서 내려받지 않습니다. 로컬 CPU 검증은 random-init
tiny config만 씁니다. 실제 screening과 4B numerical audit는 SERVER_PENDING입니다.
"""

from __future__ import annotations

import importlib
from contextlib import ExitStack
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..page import N_BODY_LANDMARKS, N_LANDMARKS, Page, provenance_hash
from . import SERVER_PENDING, AdapterUnavailable


def _transformers():
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        raise AdapterUnavailable(
            f"{SERVER_PENDING}: transformers is an optional server dependency; "
            "install the 'server' extra to use the Qwen adapter"
        ) from exc


def probe_qwen() -> dict:
    """어떤 Qwen config class를 쓸 수 있는지 보고합니다."""
    try:
        tf = _transformers()
    except AdapterUnavailable as exc:
        return {"available": False, "status": SERVER_PENDING, "reason": str(exc)}
    families = {
        "qwen3": hasattr(tf, "Qwen3Config") and hasattr(tf, "Qwen3ForCausalLM"),
        "qwen2": hasattr(tf, "Qwen2Config") and hasattr(tf, "Qwen2ForCausalLM"),
    }
    return {
        "available": any(families.values()),
        "transformers_version": tf.__version__,
        "families": families,
        "qwen3_ready": families["qwen3"],
        "status": "ok" if families["qwen3"] else SERVER_PENDING,
        "note": (
            "Qwen3-4B extraction needs transformers>=4.51. Qwen2 tiny config는 layer 구조가"
            " 같아 hook 논리를 CPU에서 검증하는 용도로만 씁니다."
        ),
    }


@dataclass
class TinyQwen:
    """random-init tiny Qwen. CPU 검증 전용이며 실제 weights가 아닙니다."""

    model: nn.Module
    family: str
    hidden_size: int
    n_layers: int


def build_tiny_qwen(
    hidden_size: int = 32, n_layers: int = 4, n_heads: int = 4, vocab_size: int = 64
) -> TinyQwen:
    """Qwen3 config가 있으면 그것을, 없으면 Qwen2 config로 tiny model을 만듭니다."""
    tf = _transformers()
    common = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 2,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_heads,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }
    if hasattr(tf, "Qwen3Config"):
        config = tf.Qwen3Config(**common)
        model = tf.Qwen3ForCausalLM(config)
        family = "qwen3"
    elif hasattr(tf, "Qwen2Config"):
        config = tf.Qwen2Config(**common)
        model = tf.Qwen2ForCausalLM(config)
        family = "qwen2"
    else:
        raise AdapterUnavailable(
            f"{SERVER_PENDING}: neither Qwen3Config nor Qwen2Config is available in transformers"
        )
    model.eval()
    return TinyQwen(model=model, family=family, hidden_size=hidden_size, n_layers=n_layers)


def _decoder_layers(model: nn.Module) -> list[nn.Module]:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AdapterUnavailable("cannot locate decoder layers on the given model")
    return list(layers)


def select_landmarks(
    candidate_offsets: list[int], prompt_len: int, n_body: int = N_BODY_LANDMARKS
) -> tuple[Tensor, Tensor, Tensor]:
    """문제 본문 landmark 16개 + canonical final prompt token 1개를 고릅니다.

    중복 offset과 부족한 자리는 validity mask로 처리합니다. 원본과 변형의 같은
    landmark 순번이 같은 수학 개념이라고 가정하지 않습니다.
    반환: (offsets [P] int64, valid [P] bool, relative_positions [P] float32)
    """
    if prompt_len < 1:
        raise ValueError("prompt_len must be >= 1")
    offsets: list[int] = []
    valid: list[bool] = []
    seen: set[int] = set()
    for offset in candidate_offsets:
        if len(offsets) >= n_body:
            break
        in_range = 0 <= offset < prompt_len
        if not in_range or offset in seen:
            offsets.append(min(max(offset, 0), prompt_len - 1))
            valid.append(False)  # 중복/범위 밖 landmark는 padding으로 처리합니다.
            continue
        seen.add(offset)
        offsets.append(offset)
        valid.append(True)
    while len(offsets) < n_body:
        offsets.append(prompt_len - 1)
        valid.append(False)
    offsets.append(prompt_len - 1)  # canonical final prompt token
    valid.append(True)
    assert len(offsets) == N_LANDMARKS
    offsets_t = torch.tensor(offsets, dtype=torch.long)
    rel = offsets_t.float() / max(prompt_len - 1, 1)
    return offsets_t, torch.tensor(valid, dtype=torch.bool), rel.float()


@torch.no_grad()
def extract_page(
    model: nn.Module,
    input_ids: Tensor,
    landmark_offsets: Tensor,
    valid: Tensor,
    relative_positions: Tensor,
    original_id: str,
    variant_id: str,
    is_identity: bool = False,
    provenance: dict | None = None,
) -> Page:
    """한 prompt를 흘려보내 Page를 만듭니다. input_ids는 [1, T]입니다."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must be [1, T], got {tuple(input_ids.shape)}")
    layers = _decoder_layers(model)
    n_layers = len(layers)
    layer_inputs: list[Tensor] = [None] * n_layers  # type: ignore[list-item]
    layer_outputs: list[Tensor] = [None] * n_layers  # type: ignore[list-item]
    attn_out: list[Tensor] = [None] * n_layers  # type: ignore[list-item]
    mlp_out: list[Tensor] = [None] * n_layers  # type: ignore[list-item]

    def pre_hook(index: int):
        def hook(_module, args, kwargs):  # noqa: ANN001
            hidden = kwargs.get("hidden_states") if kwargs else None
            if hidden is None:
                hidden = args[0]
            layer_inputs[index] = hidden.detach()[0]
            return None

        return hook

    def out_hook(store: list, index: int):
        def hook(_module, _args, output):  # noqa: ANN001
            tensor = output[0] if isinstance(output, tuple) else output
            store[index] = tensor.detach()[0]
            return None

        return hook

    with ExitStack() as stack:
        for i, layer in enumerate(layers):
            stack.callback(
                layer.register_forward_pre_hook(pre_hook(i), with_kwargs=True).remove
            )
            stack.callback(layer.register_forward_hook(out_hook(layer_outputs, i)).remove)
            stack.callback(layer.self_attn.register_forward_hook(out_hook(attn_out, i)).remove)
            stack.callback(layer.mlp.register_forward_hook(out_hook(mlp_out, i)).remove)
        model(input_ids=input_ids, use_cache=False)

    idx = landmark_offsets
    state = torch.stack(
        [layer_inputs[d][idx] for d in range(n_layers)] + [layer_outputs[-1][idx]]
    ).float()  # [L+1, P, H]
    updates = torch.stack(
        [torch.stack([attn_out[d][idx], mlp_out[d][idx]], dim=1) for d in range(n_layers)]
    ).float()  # [L, P, 2, H]

    meta = dict(provenance or {})
    meta.setdefault("extractor", "aimo.adapters.qwen.extract_page")
    meta.setdefault("n_layers", n_layers)
    meta.setdefault("prompt_len", int(input_ids.shape[1]))
    meta["config_hash"] = provenance_hash(
        getattr(getattr(model, "config", None), "to_dict", dict)()
    )
    page = Page(
        state=state,
        updates=updates,
        valid=valid,
        token_offsets=landmark_offsets,
        relative_positions=relative_positions,
        original_id=original_id,
        variant_id=variant_id,
        is_identity=is_identity,
        provenance=meta,
    )
    page.validate(tol=1e-3)  # float32 forward 오차를 감안한 residual identity 확인
    return page


def screening_ready() -> dict:
    """실제 screening 가능 여부. 로컬에서는 항상 SERVER_PENDING입니다."""
    probe = probe_qwen()
    return {
        "status": SERVER_PENDING,
        "reason": (
            "실제 screening은 Qwen3-4B weights와 GPU가 필요합니다. 로컬에서는 weights를"
            " 내려받지 않으며 numerical audit도 수행하지 않습니다."
        ),
        "qwen_probe": probe,
    }
