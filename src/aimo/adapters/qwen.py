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
import re
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


# --------------------------------------------------------------------------------------
# v2 research thinking protocol (어려운 문제용)
# --------------------------------------------------------------------------------------

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# 최종 답 제출 형식. thinking 블록 밖에서만 찾습니다.
_FINAL_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_FINAL_LABEL = re.compile(r"(?:final answer)\s*[:=]\s*(.+)", re.IGNORECASE)

SCORER_ID = "aimo.thinking.exact_final_answer"
SCORER_VERSION = "1"


def split_thinking(text: str) -> tuple[str, str, bool]:
    """(thinking, answer_region, closed)로 나눕니다.

    thinking 블록이 닫히지 않았으면 answer region은 비어 있고 closed=False입니다. 생각
    중간에 정답 숫자가 나타났다는 이유로 C 처리하지 않기 위해 answer region만 채점합니다.
    """
    if THINK_OPEN not in text and THINK_CLOSE not in text:
        return "", text, True  # thinking을 쓰지 않은 응답
    head, _, rest = text.partition(THINK_OPEN)
    thinking, closer, tail = rest.partition(THINK_CLOSE)
    if not closer:
        return thinking, "", False
    return thinking, head + tail, True


def parse_submitted_answer(answer_region: str) -> str | None:
    """answer region에서 제출된 최종 답만 뽑습니다. 없으면 None입니다.

    마지막 \\boxed{...} -> 마지막 "final answer:" 라벨 순서입니다. 임의의 마지막 숫자를
    정답으로 쓰지 않습니다 (thinking 경로에서는 오판 위험이 큽니다).
    """
    boxed = _FINAL_BOXED.findall(answer_region)
    if boxed:
        return _normalize_answer(boxed[-1])
    labelled = _FINAL_LABEL.findall(answer_region)
    if labelled:
        return _normalize_answer(labelled[-1].splitlines()[0])
    return None


def _normalize_answer(value: str) -> str:
    cleaned = value.strip().strip("$").rstrip(".").replace(",", "").replace(" ", "")
    return cleaned


def score_answer(predicted: str | None, gold: str) -> bool:
    """version pin된 exact scorer. LLM judge를 쓰지 않습니다."""
    if predicted is None:
        return False
    left, right = _normalize_answer(predicted), _normalize_answer(gold)
    try:
        return abs(float(left) - float(right)) < 1e-9
    except ValueError:
        return left == right


def check_total_context(prompt_tokens: int, generated_tokens: int, limit: int | None) -> None:
    """prompt + generated token 합계를 검사합니다.

    한도를 조용히 늘리거나 truncate하지 않고 오류로 알립니다.
    """
    if limit is None:
        raise ValueError(
            "max_total_context is not calibrated yet; set it in the server calibration "
            "manifest before running generation"
        )
    total = prompt_tokens + generated_tokens
    if total > limit:
        raise ValueError(
            f"total context {total} (prompt {prompt_tokens} + generated {generated_tokens}) "
            f"exceeds the configured limit {limit}; do not truncate or raise it silently"
        )


def classify_thinking_slot(
    *,
    started: bool,
    infra_error: bool,
    hit_cap: bool,
    is_final_cap: bool,
    text: str | None,
    gold: str,
) -> str:
    """thinking profile의 slot outcome.

    cap-hit, 운영 중단, 채점 모호를 W로 합치지 않습니다. 중간 checkpoint의 미완료(X)와
    명시된 최종 scoring deadline(is_final_cap)을 구분합니다. 최종 cap에서 실패로 볼지는
    protocol의 final_cap_failure_policy로 따로 정합니다.
    """
    from ..labels import (
        OUTCOME_CAP_HIT,
        OUTCOME_CORRECT,
        OUTCOME_INFRA_ERROR,
        OUTCOME_NOT_STARTED,
        OUTCOME_UNSCORED,
        OUTCOME_WRONG,
    )

    if not started:
        return OUTCOME_NOT_STARTED
    if infra_error:
        return OUTCOME_INFRA_ERROR
    _thinking, answer_region, closed = split_thinking(text or "")
    if not closed:
        # thinking이 닫히지 않은 미완료 generation을 completed-wrong으로 기록하지 않습니다.
        return OUTCOME_CAP_HIT
    submitted = parse_submitted_answer(answer_region)
    if submitted is None:
        return OUTCOME_CAP_HIT if hit_cap else OUTCOME_UNSCORED
    if hit_cap and not is_final_cap:
        return OUTCOME_CAP_HIT
    return OUTCOME_CORRECT if score_answer(submitted, gold) else OUTCOME_WRONG


def thinking_ready(profile) -> dict:
    """thinking profile의 calibration 상태를 보고합니다."""
    pending = profile.needs_calibration()
    return {
        "model_id": profile.model_id,
        "enable_thinking": profile.enable_thinking,
        "sampling": {
            "do_sample": profile.do_sample,
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "top_k": profile.top_k,
            "min_p": profile.min_p,
        },
        "needs_calibration": pending,
        "protocol_hash": profile.protocol_hash(),
        "scorer": {"id": SCORER_ID, "version": SCORER_VERSION},
        "status": SERVER_PENDING if pending else "calibrated",
        "note": (
            "Qwen thinking 기반 연구 profile입니다. 공식 AIMO 평가와 동일하지 않습니다."
        ),
    }
