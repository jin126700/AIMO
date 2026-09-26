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
from ..scoring import (
    SUPPORTED_FORMS,
    VERDICT_CORRECT,
    VERDICT_UNSUPPORTED,
    compare_answers,
    extract_boxed,
)
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


def _backbone(model: nn.Module) -> nn.Module:
    """decoder backbone을 돌려줍니다.

    LM head를 지나면 [T, vocab] logits이 만들어져 메모리를 크게 낭비합니다. Page 추출에는
    hidden state만 필요하므로 backbone(`model.model`)을 직접 호출합니다.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    return model


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
    """한 prompt를 흘려보내 Page를 만듭니다. input_ids는 [1, T]입니다.

    메모리 규칙: hook 안에서 **필요한 landmark만 골라 CPU float32로 복사**합니다. 모든
    layer의 전체 [T, H] activation을 들고 있지 않고, LM head도 지나지 않습니다.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must be [1, T], got {tuple(input_ids.shape)}")
    layers = _decoder_layers(model)
    n_layers = len(layers)
    device = input_ids.device
    index = landmark_offsets.to(device)
    # 필요한 landmark만 담는 작은 buffer들 ([P, H] × layer).
    layer_inputs: list[Tensor | None] = [None] * n_layers
    layer_outputs: list[Tensor | None] = [None] * n_layers
    attn_out: list[Tensor | None] = [None] * n_layers
    mlp_out: list[Tensor | None] = [None] * n_layers

    def take(tensor: Tensor) -> Tensor:
        """[1, T, H]에서 landmark만 골라 CPU float32로 복사합니다."""
        return tensor.detach()[0].index_select(0, index).to("cpu", torch.float32).clone()

    def pre_hook(target: list, position: int):
        def hook(_module, args, kwargs):  # noqa: ANN001
            hidden = kwargs.get("hidden_states") if kwargs else None
            if hidden is None:
                hidden = args[0]
            target[position] = take(hidden)
            return None

        return hook

    def out_hook(target: list, position: int):
        def hook(_module, _args, output):  # noqa: ANN001
            tensor = output[0] if isinstance(output, tuple) else output
            target[position] = take(tensor)
            return None

        return hook

    with ExitStack() as stack:
        for i, layer in enumerate(layers):
            stack.callback(
                layer.register_forward_pre_hook(
                    pre_hook(layer_inputs, i), with_kwargs=True
                ).remove
            )
            stack.callback(layer.register_forward_hook(out_hook(layer_outputs, i)).remove)
            stack.callback(layer.self_attn.register_forward_hook(out_hook(attn_out, i)).remove)
            stack.callback(layer.mlp.register_forward_hook(out_hook(mlp_out, i)).remove)
        # LM head를 지나지 않도록 backbone을 직접 호출합니다.
        _backbone(model)(input_ids=input_ids, use_cache=False)

    missing = [i for i in range(n_layers) if layer_inputs[i] is None or mlp_out[i] is None]
    if missing:
        raise RuntimeError(f"hooks did not fire for layer(s) {missing}")
    state = torch.stack(
        [layer_inputs[d] for d in range(n_layers)] + [layer_outputs[-1]]
    )  # [L+1, P, H]
    updates = torch.stack(
        [torch.stack([attn_out[d], mlp_out[d]], dim=1) for d in range(n_layers)]
    )  # [L, P, 2, H]

    meta = dict(provenance or {})
    meta.setdefault("extractor", "aimo.adapters.qwen.extract_page")
    meta.setdefault("n_layers", n_layers)
    meta.setdefault("prompt_len", int(input_ids.shape[1]))
    meta.setdefault("dtype", "float32")
    meta.setdefault("backend", type(_backbone(model)).__name__)
    meta["config_hash"] = provenance_hash(
        getattr(getattr(model, "config", None), "to_dict", dict)()
    )
    page = Page(
        state=state,
        updates=updates,
        valid=valid.cpu(),
        token_offsets=landmark_offsets.cpu(),
        relative_positions=relative_positions.cpu(),
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

# 제출된 최종 답 라벨. thinking 블록 밖에서만 찾습니다.
_FINAL_LABEL = re.compile(r"(?:final answer)\s*[:=]\s*(.+)", re.IGNORECASE)

SCORER_ID = "aimo.thinking.exact_final_answer"
# v1은 float 비교와 단일 brace boxed parser를 썼습니다. v2는 exact Fraction 비교와
# nested brace parser를 씁니다. 이전 결과를 v2 결과로 덮어쓰지 않습니다.
SCORER_VERSION = "2"

# split_thinking의 상태.
SPLIT_NO_THINKING = "no_thinking"  # tag가 전혀 없고 prompt도 열지 않음 -> 전체가 답 영역
SPLIT_CLOSED = "closed"  # thinking이 닫힘 -> 닫힌 뒤의 내용만 답 영역
SPLIT_UNCLOSED = "unclosed"  # thinking이 열린 채 끝남 -> 제출된 답 없음
SPLIT_EMPTY = "empty"  # 의미 있는 생성이 없음 (special token만 또는 공백)

# 제출된 답의 상태.
SUBMIT_NONE = "none"
SUBMIT_SINGLE = "single"
SUBMIT_CONFLICTING = "conflicting"


@dataclass
class ThinkingSplit:
    """generated text를 thinking과 답 영역으로 나눈 결과."""

    thinking: str
    answer_region: str
    state: str

    @property
    def closed(self) -> bool:
        return self.state in (SPLIT_CLOSED, SPLIT_NO_THINKING)


def split_thinking(
    text: str | None,
    *,
    thinking_already_open: bool = False,
    generated_token_ids: list[int] | None = None,
    special_token_ids: frozenset[int] | set[int] = frozenset(),
) -> ThinkingSplit:
    """generated text를 (thinking, answer_region, state)로 나눕니다.

    generated text에 `<think>`가 반드시 있다고 가정하지 않습니다. chat template이 이미
    thinking을 열어 둔 경우(`thinking_already_open=True`)에는 generated suffix에 `</think>`만
    있을 수 있습니다. tag 상태와 생성된 token ID를 함께 고려합니다.

    규칙:
      - `thinking_already_open` 또는 `<think>`로 열린 뒤 `</think>`가 나오면 닫힌 것으로 보고,
        닫힌 뒤(그리고 열리기 전)의 내용만 답 영역으로 씁니다.
      - `</think>`가 `<think>`보다 먼저 나오면 prompt가 열어 둔 것으로 보고 같은 규칙을
        적용합니다 (generated text에 `<think>`가 있다고 가정하지 않습니다).
      - 여러 개의 thinking 블록이 있어도 같은 규칙을 반복 적용합니다.
      - 끝까지 열려 있으면 `unclosed`이고 답 영역은 비어 있습니다. tag가 없다는 이유만으로
        thinking 중간 내용을 최종 답으로 채점하지 않습니다.
      - 생성 token이 없거나 special token뿐이면 `empty`입니다.
    """
    body = text or ""
    if generated_token_ids is not None:
        meaningful = [tid for tid in generated_token_ids if tid not in special_token_ids]
        if not meaningful:
            # special token만 생성하고 중단된 상태. 제출된 답이 없습니다.
            return ThinkingSplit(thinking=body.strip(), answer_region="", state=SPLIT_EMPTY)

    # `</think>`가 어떤 `<think>`보다 먼저 나오면 prompt/template이 thinking을 열어 둔
    # 것입니다. 호출자가 flag를 주지 않아도 이 사실만으로 열린 상태로 봅니다.
    close_at = body.find(THINK_CLOSE)
    open_at = body.find(THINK_OPEN)
    if close_at != -1 and (open_at == -1 or close_at < open_at):
        thinking_already_open = True

    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    is_open = thinking_already_open
    saw_tag = thinking_already_open
    cursor = 0
    while cursor < len(body):
        target = THINK_CLOSE if is_open else THINK_OPEN
        found = body.find(target, cursor)
        if found == -1:
            (thinking_parts if is_open else answer_parts).append(body[cursor:])
            break
        (thinking_parts if is_open else answer_parts).append(body[cursor:found])
        cursor = found + len(target)
        is_open = not is_open
        saw_tag = True

    thinking = "".join(thinking_parts)
    answer_region = "".join(answer_parts)
    if is_open:
        # 열린 채 끝났으므로 답 영역을 쓰지 않습니다.
        return ThinkingSplit(thinking=thinking, answer_region="", state=SPLIT_UNCLOSED)
    if not saw_tag:
        if not body.strip():
            return ThinkingSplit(thinking="", answer_region="", state=SPLIT_EMPTY)
        return ThinkingSplit(thinking="", answer_region=body, state=SPLIT_NO_THINKING)
    if not answer_region.strip() and not thinking.strip():
        return ThinkingSplit(thinking=thinking, answer_region="", state=SPLIT_EMPTY)
    return ThinkingSplit(thinking=thinking, answer_region=answer_region, state=SPLIT_CLOSED)


@dataclass
class SubmittedAnswer:
    """답 영역에서 추출한 제출 답."""

    raw: str | None
    status: str
    candidates: list[str]


def parse_submitted_answer(answer_region: str) -> SubmittedAnswer:
    r"""답 영역에서 제출된 최종 답만 뽑습니다.

    우선순위는 `oxed{...}` -> `final answer:` 라벨입니다. 임의의 마지막 숫자를 답으로
    쓰지 않습니다 (thinking 경로에서 오판 위험이 큽니다).

    **복수의 상충하는 답 규칙**: 같은 우선순위 안에서 후보가 여러 개이면 지원 범위에서
    모두 동치일 때만 채택하고(마지막 값을 raw로 씁니다), 하나라도 다르거나 비교할 수 없으면
    `conflicting`으로 보고합니다. conflicting은 오답이 아니라 채점 불가입니다.
    """
    for candidates in (
        extract_boxed(answer_region),
        [item.splitlines()[0] for item in _FINAL_LABEL.findall(answer_region)],
    ):
        cleaned = [item for item in (value.strip() for value in candidates) if item]
        if not cleaned:
            continue
        if len(cleaned) == 1:
            return SubmittedAnswer(raw=cleaned[0], status=SUBMIT_SINGLE, candidates=cleaned)
        last = cleaned[-1]
        if all(compare_answers(item, last) == VERDICT_CORRECT for item in cleaned[:-1]):
            return SubmittedAnswer(raw=last, status=SUBMIT_SINGLE, candidates=cleaned)
        return SubmittedAnswer(raw=None, status=SUBMIT_CONFLICTING, candidates=cleaned)
    return SubmittedAnswer(raw=None, status=SUBMIT_NONE, candidates=[])


def score_answer(predicted: str | None, gold: str) -> str:
    """exact scorer. correct / wrong / unsupported를 돌려줍니다.

    지원 범위와 정규화 규칙은 `aimo.scoring`에 있습니다. 임의 eval을 하지 않고 LLM judge를
    쓰지 않습니다.
    """
    return compare_answers(predicted, gold)


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
    thinking_already_open: bool = False,
    generated_token_ids: list[int] | None = None,
    special_token_ids: frozenset[int] | set[int] = frozenset(),
) -> str:
    """thinking profile의 slot outcome.

    미완료(X) / 채점 불가(U_score) / 명확한 오답(W)을 구분합니다. cap-hit, 운영 중단, 채점
    모호를 W로 합치지 않습니다. 중간 checkpoint의 미완료와 명시된 최종 scoring
    deadline(`is_final_cap`)을 구분하며, 최종 cap에서 실패로 볼지는 protocol의
    `final_cap_failure_policy`로 따로 정합니다.
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
    split = split_thinking(
        text,
        thinking_already_open=thinking_already_open,
        generated_token_ids=generated_token_ids,
        special_token_ids=special_token_ids,
    )
    if split.state in (SPLIT_UNCLOSED, SPLIT_EMPTY):
        # 미완료 generation을 completed-wrong으로 기록하지 않습니다.
        return OUTCOME_CAP_HIT
    submitted = parse_submitted_answer(split.answer_region)
    if submitted.status == SUBMIT_NONE:
        return OUTCOME_CAP_HIT if hit_cap else OUTCOME_UNSCORED
    if submitted.status == SUBMIT_CONFLICTING:
        # 상충하는 최종 답은 오답이 아니라 채점 불가입니다.
        return OUTCOME_UNSCORED
    if hit_cap and not is_final_cap:
        return OUTCOME_CAP_HIT
    verdict = score_answer(submitted.raw, gold)
    if verdict == VERDICT_UNSUPPORTED:
        return OUTCOME_UNSCORED
    return OUTCOME_CORRECT if verdict == VERDICT_CORRECT else OUTCOME_WRONG


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
        "scorer": {
            "id": SCORER_ID,
            "version": SCORER_VERSION,
            "supported_forms": list(SUPPORTED_FORMS),
        },
        "status": SERVER_PENDING if pending else "calibrated",
        "note": (
            "Qwen thinking 기반 연구 profile입니다. 공식 AIMO 평가와 동일하지 않습니다."
        ),
    }


# --------------------------------------------------------------------------------------
# Generation backend (dependency injection)
# --------------------------------------------------------------------------------------


@dataclass
class SlotRequest:
    """slot 하나에 대한 generation 요청."""

    prompt_id: str
    slot_id: str
    prompt: str
    gold: str
    seed: int
    is_final_cap: bool = False
    # chat template이 이미 thinking을 열어 둔 경우 True입니다.
    thinking_already_open: bool = False


@dataclass
class SlotResult:
    """slot 하나의 generation 결과. 채점은 하지 않습니다."""

    slot_id: str
    started: bool = True
    infra_error: bool = False
    text: str | None = None
    hit_cap: bool = False
    prompt_tokens: int = 0
    generated_tokens: int = 0
    generated_token_ids: list[int] | None = None
    thinking_already_open: bool = False
    error_message: str | None = None


class MockGenerationBackend:
    """로컬 검증용 backend. 실제 weights를 로드하거나 generate하지 않습니다.

    `responses`는 slot_id -> SlotResult 매핑입니다. 없는 slot은 not_started로 돌려줍니다.
    """

    def __init__(self, responses: dict[str, SlotResult], special_token_ids: set[int] | None = None):
        self.responses = responses
        self.special_token_ids = frozenset(special_token_ids or set())
        self.calls: list[str] = []

    def generate(self, request: SlotRequest) -> SlotResult:
        self.calls.append(request.slot_id)
        result = self.responses.get(request.slot_id)
        if result is None:
            return SlotResult(slot_id=request.slot_id, started=False)
        return result


class QwenGenerationBackend:
    """실제 Qwen weights를 쓰는 backend.

    **로컬에서는 인스턴스화하지 않습니다.** 서버에서 확정된 calibration 값과 실제 weights가
    있을 때만 씁니다. 여기서는 필요한 config 검증과 호출 형태만 준비합니다.
    """

    def __init__(self, profile, model=None, tokenizer=None) -> None:
        pending = profile.needs_calibration()
        if pending:
            raise AdapterUnavailable(
                f"{SERVER_PENDING}: thinking profile is not calibrated yet ({pending}); "
                "fix them in the server calibration manifest before loading weights"
            )
        if model is None or tokenizer is None:
            raise AdapterUnavailable(
                f"{SERVER_PENDING}: real Qwen weights and tokenizer must be supplied on the "
                "server; this repository does not download 4B weights locally"
            )
        self.profile = profile
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, request: SlotRequest) -> SlotResult:  # pragma: no cover - 서버 전용
        raise AdapterUnavailable(
            f"{SERVER_PENDING}: real generation runs on the server; verified locally only "
            "through MockGenerationBackend"
        )


# --------------------------------------------------------------------------------------
# Page extraction runner
# --------------------------------------------------------------------------------------


@dataclass
class ExtractionRequest:
    """Page 추출 요청 하나."""

    original_id: str
    variant_id: str
    input_ids: Tensor  # [1, T]
    landmark_offsets: Tensor
    valid: Tensor
    relative_positions: Tensor
    is_identity: bool = False


def run_page_extraction(
    model: nn.Module,
    requests: list[ExtractionRequest],
    *,
    ledger=None,
    guard=None,
    provenance: dict | None = None,
    tolerance: float = 1e-3,
) -> tuple[list[Page], dict]:
    """요청 목록을 실제로 추출합니다. tiny model과 실제 weights가 같은 경로를 씁니다.

    dedup/resume은 `variant_id` 기준이며, 예산 초과나 중단 요청이 오면 남은 요청을 건너뛰고
    부분 결과를 돌려줍니다. hook 안에서 landmark만 CPU로 복사하므로 layer 수에 비례해
    전체 activation을 들고 있지 않습니다.
    """
    pages: list[Page] = []
    report = {"extracted": 0, "skipped": 0, "stopped": False, "stop_reason": "", "errors": []}
    for request in requests:
        if ledger is not None and ledger.seen(request.variant_id):
            report["skipped"] += 1
            continue
        if guard is not None:
            hit, why = guard.should_stop()
            if hit:
                report["stopped"] = True
                report["stop_reason"] = why
                break
        try:
            page = extract_page(
                model,
                request.input_ids,
                request.landmark_offsets,
                request.valid,
                request.relative_positions,
                request.original_id,
                request.variant_id,
                is_identity=request.is_identity,
                provenance=provenance,
            )
        except (ValueError, RuntimeError) as exc:
            report["errors"].append(f"{request.variant_id}: {type(exc).__name__}: {exc}")
            continue
        error = page.residual_identity_error()
        if error > tolerance:
            report["errors"].append(
                f"{request.variant_id}: residual identity error {error:.3e} > {tolerance:.3e}"
            )
            continue
        pages.append(page)
        report["extracted"] += 1
        if ledger is not None:
            ledger.mark(request.variant_id, {"residual_identity_error": error})
    return pages, report


def load_real_qwen(cfg) -> tuple[nn.Module, object]:
    """실제 Qwen weights와 tokenizer를 로드합니다 (서버 전용).

    **이 저장소는 로컬에서 4B weights를 내려받지 않습니다.** 로컬에서 호출하면
    SERVER_PENDING으로 fail-fast하며, 추출 경로 자체는 tiny model로 검증합니다.
    """
    tf = _transformers()
    profile = cfg.server.thinking
    pending = profile.needs_calibration()
    if pending:
        raise AdapterUnavailable(
            f"{SERVER_PENDING}: thinking profile is not calibrated yet ({pending})"
        )
    if not hasattr(tf, "Qwen3ForCausalLM"):
        raise AdapterUnavailable(
            f"{SERVER_PENDING}: transformers {tf.__version__} has no Qwen3 support; "
            "Qwen3-4B needs transformers>=4.51 on the server"
        )
    raise AdapterUnavailable(
        f"{SERVER_PENDING}: loading {profile.model_id!r} weights is a server step; this "
        "repository does not download them locally"
    )
