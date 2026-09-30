"""고정 macro × all-token native extractor.

decoder layer `l`의 actual residual update를 다음처럼 정의합니다.

    M_l = h_mid - h_in      (Mixer가 residual에 실제로 더한 값)
    F_l = h_out - h_mid     (FFN이 residual에 실제로 더한 값)

`h_mid`는 layout이 정한 module의 입력에서 읽습니다 (`model_registry`). linear-attention
Mixer와 full-attention Mixer가 섞인 hybrid model에서도 residual에 더해진 값 자체를 쓰므로
Mixer 내부 gate나 출력 형식에 의존하지 않습니다. Mixer / FFN module의 hook 출력은 이 값과
일치하는지 확인하는 audit 값으로만 기록합니다.

macro g (`b_g <= l < b_{g+1}`)에 대해 `M_g = Σ M_l`, `F_g = Σ F_l`이고
`S_{b_{g+1}} - S_{b_g} ≈ M_g + F_g`를 검사합니다.

- raw final boundary(`S_L`)는 마지막 decoder layer의 출력 hook에서 읽고, final norm 이후
  hidden은 backbone `norm` module의 출력 hook에서 따로 읽습니다. `output_hidden_states`의
  마지막 항목을 raw final boundary로 가정하지 않습니다.
- 모든 token을 유지합니다. landmark sampling, projection 전 mean pooling, 무단 truncation을
  하지 않습니다. 가변 길이는 오른쪽 padding과 `attention_mask`로 처리합니다.
- `AuditSink`는 offline audit용으로 완전한 State / M / F를 CPU float32로 보관합니다.
  `StreamSink`는 제출 경로용으로 update가 생길 때마다 저차원으로 누적하고 raw Page를
  보관하지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .macro_page import macro_boundaries
from .model_registry import MIXER_KIND_BY_ATTR, LayoutSpec, layout_for_model_type
from .native_page import (
    SEG_PREFIX,
    SEG_PROBE,
    SEG_PROBLEM,
    SEG_TEMPLATE,
    DataLimit,
    NativePage,
    NativeProvenance,
    SchemaIncompatible,
)

# backbone(`layers`와 final `norm`을 가진 module)을 찾는 확인된 경로.
BACKBONE_PATHS = ("model", "model.language_model", "language_model.model", "language_model", "")


def _get_path(root: nn.Module, path: str) -> nn.Module | None:
    node = root
    for part in [p for p in path.split(".") if p]:
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


@dataclass
class Backbone:
    module: nn.Module
    layers: list[nn.Module]
    final_norm: nn.Module
    path: str


def locate_backbone(model: nn.Module) -> Backbone:
    for path in BACKBONE_PATHS:
        node = _get_path(model, path)
        if node is None:
            continue
        layers = getattr(node, "layers", None)
        norm = getattr(node, "norm", None)
        if isinstance(layers, nn.ModuleList) and isinstance(norm, nn.Module):
            return Backbone(module=node, layers=list(layers), final_norm=norm, path=path)
    raise SchemaIncompatible(
        "SCHEMA_INCOMPATIBLE: cannot find a backbone with `layers` and a final `norm` on "
        f"{type(model).__name__}"
    )


def text_config(model: nn.Module):
    config = getattr(model, "config", None)
    if config is None:
        raise SchemaIncompatible("model has no config")
    return getattr(config, "text_config", None) or config


def resolve_layout(model: nn.Module) -> LayoutSpec:
    config = text_config(model)
    return layout_for_model_type(str(getattr(config, "model_type", "")))


def mixer_kinds(layers: list[nn.Module], layout: LayoutSpec) -> list[str]:
    kinds = []
    for index, layer in enumerate(layers):
        present = [name for name in ("linear_attn", "self_attn") if hasattr(layer, name)]
        if len(present) != 1:
            raise SchemaIncompatible(
                f"layer {index} must have exactly one Mixer module, found {present}"
            )
        for name in (layout.mid_module, layout.ffn_update_module):
            if not hasattr(layer, name):
                raise SchemaIncompatible(f"layer {index} has no `{name}` for layout {layout.name}")
        kinds.append(MIXER_KIND_BY_ATTR[present[0]])
    return kinds


def _first_tensor(value) -> Tensor:
    return value[0] if isinstance(value, tuple | list) else value


def _hidden_arg(args: tuple, kwargs: dict) -> Tensor:
    hidden = kwargs.get("hidden_states") if kwargs else None
    return args[0] if hidden is None else hidden


class ExtractionSink:
    """extractor가 관측을 넘기는 곳. audit와 streaming이 같은 hook 논리를 공유합니다."""

    def on_boundary(self, g: int, hidden: Tensor) -> None:  # S_{b_g}, [B,T,H]
        raise NotImplementedError

    def on_layer_update(self, g: int, mixer: Tensor, ffn: Tensor) -> None:
        raise NotImplementedError

    def on_module_update(self, g: int, mixer: Tensor, ffn: Tensor) -> None:
        pass

    def on_final_norm(self, hidden: Tensor) -> None:
        pass


class AuditSink(ExtractionSink):
    """완전한 State / M / F를 CPU float32로 보관하는 offline audit sink."""

    def __init__(self, n_macro: int, device: str | torch.device = "cpu",
                 states_only: bool = False) -> None:
        self.n_macro = n_macro
        self.device = torch.device(device)
        self.states_only = states_only
        self.state: list[Tensor | None] = [None] * (n_macro + 1)
        self.mixer: list[Tensor | None] = [None] * n_macro
        self.ffn: list[Tensor | None] = [None] * n_macro
        self.mixer_hook: list[Tensor | None] = [None] * n_macro
        self.ffn_hook: list[Tensor | None] = [None] * n_macro
        self.final_norm: Tensor | None = None

    def _add(self, slot: list, g: int, value: Tensor) -> None:
        value = value.detach().to(self.device, torch.float32)
        slot[g] = value.clone() if slot[g] is None else slot[g] + value

    def on_boundary(self, g: int, hidden: Tensor) -> None:
        self.state[g] = hidden.detach().to(self.device, torch.float32).clone()

    def on_layer_update(self, g: int, mixer: Tensor, ffn: Tensor) -> None:
        if not self.states_only:
            self._add(self.mixer, g, mixer)
            self._add(self.ffn, g, ffn)

    def on_module_update(self, g: int, mixer: Tensor, ffn: Tensor) -> None:
        self._add(self.mixer_hook, g, mixer)
        self._add(self.ffn_hook, g, ffn)

    def on_final_norm(self, hidden: Tensor) -> None:
        self.final_norm = hidden.detach().to(self.device, torch.float32).clone()

    def stacked(self) -> dict[str, Tensor]:
        missing = [i for i, value in enumerate(self.state) if value is None]
        if missing:
            raise RuntimeError(f"boundary hooks did not fire for macro boundary {missing}")
        out = {"state": torch.stack(self.state, dim=1)}  # [B, G+1, T, H]
        for name in ("mixer", "ffn", "mixer_hook", "ffn_hook"):
            values = getattr(self, name)
            if all(value is not None for value in values):
                out[name] = torch.stack(values, dim=1)
        return out


class StreamSink(ExtractionSink):
    """update가 생길 때 바로 `E_g`로 투영해 누적합니다. raw [T, H] 관측을 보관하지 않습니다.

    `encoder`는 [G, r, H], `offset`은 [G, r]. `response`는 final norm 이후 hidden을 받아
    native response sketch [B, T, q]를 만드는 함수입니다.
    """

    def __init__(
        self,
        encoder: Tensor,
        offset: Tensor,
        response: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.encoder = encoder
        self.offset = offset
        self.response = response
        g = encoder.shape[0]
        self.z_state: list[Tensor | None] = [None] * g
        self.z_in: list[Tensor | None] = [None] * g
        self.z_mixer: list[Tensor | None] = [None] * g
        self.z_ffn: list[Tensor | None] = [None] * g
        self.y: Tensor | None = None
        self.final_norm_last: Tensor | None = None
        self.last_index: Tensor | None = None

    def _project(self, g: int, hidden: Tensor) -> Tensor:
        e = self.encoder[g].to(hidden.device)
        return hidden.float() @ e.T

    def on_boundary(self, g: int, hidden: Tensor) -> None:
        n_macro = self.encoder.shape[0]
        if g >= 1:
            offset = self.offset[g - 1].to(hidden.device)
            self.z_state[g - 1] = self._project(g - 1, hidden) + offset
        if g < n_macro:
            self.z_in[g] = self._project(g, hidden) + self.offset[g].to(hidden.device)

    def on_layer_update(self, g: int, mixer: Tensor, ffn: Tensor) -> None:
        zm, zf = self._project(g, mixer), self._project(g, ffn)
        self.z_mixer[g] = zm if self.z_mixer[g] is None else self.z_mixer[g] + zm
        self.z_ffn[g] = zf if self.z_ffn[g] is None else self.z_ffn[g] + zf

    def on_final_norm(self, hidden: Tensor) -> None:
        if self.response is not None:
            self.y = self.response(hidden.float())
        if self.last_index is not None:
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            self.final_norm_last = hidden[rows, self.last_index.to(hidden.device)].float()

    def stacked(self) -> dict[str, Tensor]:
        return {
            "z_state": torch.stack(self.z_state, dim=1),  # [B, G, T, r]
            "z_in": torch.stack(self.z_in, dim=1),
            "z_mixer": torch.stack(self.z_mixer, dim=1),
            "z_ffn": torch.stack(self.z_ffn, dim=1),
        }


def run_extraction(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    sink: ExtractionSink,
    n_macro: int,
    *,
    audit_modules: bool = False,
) -> dict:
    """hook을 걸고 backbone을 한 번 흘립니다. LM head는 지나지 않습니다."""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must both be [B, T]")
    layout = resolve_layout(model)
    backbone = locate_backbone(model)
    layers = backbone.layers
    kinds = mixer_kinds(layers, layout)
    bounds = macro_boundaries(len(layers), n_macro)
    stage_of = [g for g in range(n_macro) for _ in range(bounds[g], bounds[g + 1])]
    h_in: dict[int, Tensor] = {}
    h_mid: dict[int, Tensor] = {}
    hook_mixer: dict[int, Tensor] = {}
    hook_ffn: dict[int, Tensor] = {}
    fired = {"final_norm": False}

    def layer_pre(index: int):
        def hook(_module, args, kwargs):  # noqa: ANN001
            hidden = _hidden_arg(args, kwargs)
            h_in[index] = hidden.detach()
            if index == bounds[stage_of[index]]:
                sink.on_boundary(stage_of[index], hidden.detach())

        return hook

    def mid_pre(index: int):
        def hook(_module, args, kwargs):  # noqa: ANN001
            h_mid[index] = _hidden_arg(args, kwargs).detach()

        return hook

    def layer_post(index: int):
        def hook(_module, _args, output):  # noqa: ANN001
            h_out = _first_tensor(output).detach()
            g = stage_of[index]
            start, mid = h_in.pop(index), h_mid.pop(index)
            sink.on_layer_update(g, mid.float() - start.float(), h_out.float() - mid.float())
            if audit_modules:
                sink.on_module_update(g, hook_mixer.pop(index).float(), hook_ffn.pop(index).float())
            if index + 1 == bounds[-1]:
                sink.on_boundary(n_macro, h_out)

        return hook

    def module_out(store: dict, index: int):
        def hook(_module, _args, output):  # noqa: ANN001
            store[index] = _first_tensor(output).detach()

        return hook

    def norm_out(_module, _args, output):  # noqa: ANN001
        fired["final_norm"] = True
        sink.on_final_norm(_first_tensor(output).detach())

    with ExitStack() as stack:
        for index, layer in enumerate(layers):
            reg = stack.callback
            reg(layer.register_forward_pre_hook(layer_pre(index), with_kwargs=True).remove)
            mid_module = getattr(layer, layout.mid_module)
            reg(mid_module.register_forward_pre_hook(mid_pre(index), with_kwargs=True).remove)
            if audit_modules:
                mixer_attr = next(a for a in layout.mixer_update_modules if hasattr(layer, a))
                mixer_module = getattr(layer, mixer_attr)
                reg(mixer_module.register_forward_hook(module_out(hook_mixer, index)).remove)
                ffn_module = getattr(layer, layout.ffn_update_module)
                reg(ffn_module.register_forward_hook(module_out(hook_ffn, index)).remove)
            reg(layer.register_forward_hook(layer_post(index)).remove)
        stack.callback(backbone.final_norm.register_forward_hook(norm_out).remove)
        backbone.module(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if h_in or h_mid:
        raise RuntimeError("layer hooks did not complete for every decoder layer")
    if not fired["final_norm"]:
        raise RuntimeError("final norm hook did not fire")
    return {"layout": layout.name, "mixer_kinds": kinds, "boundaries": bounds,
            "path": backbone.path}


@dataclass(frozen=True)
class PromptPolicy:
    """chat template과 effort 정책. 모든 Page/checkpoint provenance에 hash로 남깁니다."""

    effort: str = "default"
    template_kwargs: tuple[tuple[str, object], ...] = ()
    system_prompt: str | None = None
    effort_policy: str = "metadata_only"
    max_tokens: int = 4096

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["template_kwargs"] = [list(item) for item in self.template_kwargs]
        return payload

    def hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @classmethod
    def from_dict(cls, payload: dict) -> PromptPolicy:
        payload = dict(payload)
        items = payload.get("template_kwargs", ())
        payload["template_kwargs"] = tuple(tuple(item) for item in items)
        return cls(**payload)


@dataclass
class RenderedPrompt:
    text: str
    problem_start: int
    problem_end: int
    prefix_start: int | None
    suffix_segment: int = SEG_PREFIX


def chat_template_hash(tokenizer) -> str | None:
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None
    return hashlib.sha256(str(template).encode()).hexdigest()[:16]


def render_prompt(
    tokenizer,
    problem: str,
    policy: PromptPolicy,
    native_prefix: str | None = None,
    probe_text: str | None = None,
) -> RenderedPrompt:
    """문제만 user message로 넣고 generation prompt를 엽니다. 문제 text는 수정하지 않습니다.

    generation prompt 뒤에는 target model의 native prefix 또는 contrast probe continuation
    하나만 붙일 수 있습니다. dataset이 제공한 solution은 여기에 오지 않습니다.
    """
    if native_prefix and probe_text:
        raise ValueError("use either a native prefix or a probe continuation, not both")
    messages = []
    if policy.system_prompt:
        messages.append({"role": "system", "content": policy.system_prompt})
    messages.append({"role": "user", "content": problem})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **dict(policy.template_kwargs)
    )
    start = text.rfind(problem)
    if start < 0:
        raise SchemaIncompatible("chat template did not keep the problem text verbatim")
    prefix_start, segment = None, SEG_PREFIX
    suffix = native_prefix or probe_text
    if suffix:
        prefix_start = len(text)
        text = text + suffix
        segment = SEG_PREFIX if native_prefix else SEG_PROBE
    return RenderedPrompt(text, start, start + len(problem), prefix_start, segment)


def tokenize_prompt(tokenizer, rendered: RenderedPrompt, max_tokens: int) -> dict[str, Tensor]:
    """모든 token과 span map을 만듭니다. 길이가 넘치면 자르지 않고 DATA_LIMIT입니다."""
    encoded = tokenizer(rendered.text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise SchemaIncompatible("tokenizer does not return offset mapping; span map needs it")
    ids = list(encoded["input_ids"])
    if len(ids) > max_tokens:
        raise DataLimit(
            f"DATA_LIMIT: prompt has {len(ids)} tokens > max_tokens={max_tokens}; "
            "the native path does not truncate"
        )
    segments, spans = [], []
    for begin, end in offsets:
        if rendered.problem_start <= begin and end <= rendered.problem_end and end > begin:
            segments.append(SEG_PROBLEM)
            spans.append((begin - rendered.problem_start, end - rendered.problem_start))
        elif rendered.prefix_start is not None and begin >= rendered.prefix_start:
            segments.append(rendered.suffix_segment)
            spans.append((begin - rendered.prefix_start, end - rendered.prefix_start))
        else:
            segments.append(SEG_TEMPLATE)
            spans.append((-1, -1))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "segments": torch.tensor(segments, dtype=torch.long),
        "char_spans": torch.tensor(spans, dtype=torch.long).reshape(-1, 2),
    }


def pad_right(sequences: list[Tensor], pad_id: int) -> tuple[Tensor, Tensor]:
    t_max = max(int(seq.shape[0]) for seq in sequences)
    ids = torch.full((len(sequences), t_max), pad_id, dtype=torch.long)
    mask = torch.zeros(len(sequences), t_max, dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, : seq.shape[0]] = seq
        mask[i, : seq.shape[0]] = 1
    return ids, mask


def model_provenance(
    model: nn.Module,
    tokenizer,
    n_macro: int,
    policy: PromptPolicy,
    *,
    model_id: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    dataset: dict | None = None,
) -> NativeProvenance:
    config = text_config(model)
    layout = resolve_layout(model)
    backbone = locate_backbone(model)
    param = next(model.parameters())
    return NativeProvenance(
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_id=getattr(tokenizer, "name_or_path", model_id),
        tokenizer_revision=tokenizer_revision,
        n_layers=len(backbone.layers),
        hidden_size=int(config.hidden_size),
        vocab_size=int(config.vocab_size),
        macro_boundaries=macro_boundaries(len(backbone.layers), n_macro),
        layout=layout.name,
        mixer_kinds=mixer_kinds(backbone.layers, layout),
        dtype=str(param.dtype).replace("torch.", ""),
        backend=getattr(config, "_attn_implementation", None) or type(backbone.module).__name__,
        chat_template_hash=chat_template_hash(tokenizer),
        prompt_policy=policy.as_dict(),
        padding_side="right",
        dataset=dict(dataset or {}),
    )


@torch.no_grad()
def extract_native_pages(
    model: nn.Module,
    tokenizer,
    problems: list[dict],
    provenance: NativeProvenance,
    policy: PromptPolicy,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 1,
    tol: float | None = None,
) -> tuple[list[NativePage], dict]:
    """offline audit: 문제마다 완전한 all-token NativePage를 만들고 audit 수치를 돌려줍니다.

    `problems` 원소는 `problem_id`, `root_id`, `split`, `text`, 선택적 `native_prefix`를
    가집니다. label·answer·solution field는 받지 않습니다.
    """
    from .native_data import assert_label_free

    pages: list[NativePage] = []
    audit = {"module_agreement_mixer": [], "module_agreement_ffn": [], "identity_error": [],
             "final_norm_distinct": []}
    n_macro = provenance.n_macro
    for start in range(0, len(problems), batch_size):
        chunk = problems[start : start + batch_size]
        encoded = []
        for record in chunk:
            assert_label_free(record)
            rendered = render_prompt(tokenizer, record["text"], policy,
                                     record.get("native_prefix"), record.get("probe_text"))
            encoded.append(tokenize_prompt(tokenizer, rendered, policy.max_tokens))
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        ids, mask = pad_right([e["input_ids"] for e in encoded], pad_id)
        sink = AuditSink(n_macro)
        run_extraction(model, ids.to(device), mask.to(device), sink, n_macro, audit_modules=True)
        out = sink.stacked()
        for i, (record, enc) in enumerate(zip(chunk, encoded, strict=True)):
            t = int(enc["input_ids"].shape[0])
            page = NativePage(
                problem_id=str(record["problem_id"]),
                root_id=str(record["root_id"]),
                split=str(record["split"]),
                token_ids=enc["input_ids"],
                segments=enc["segments"],
                char_spans=enc["char_spans"],
                state=out["state"][i, :, :t].contiguous(),
                mixer=out["mixer"][i, :, :t].contiguous(),
                ffn=out["ffn"][i, :, :t].contiguous(),
                final_norm=sink.final_norm[i, :t].contiguous(),
                provenance=provenance,
                text_hash=hashlib.sha256(record["text"].strip().encode()).hexdigest()[:16],
            )
            page.validate(tol)
            agree_m = _relative(out["mixer_hook"][i, :, :t], page.mixer)
            agree_f = _relative(out["ffn_hook"][i, :, :t], page.ffn)
            audit["module_agreement_mixer"].append(agree_m)
            audit["module_agreement_ffn"].append(agree_f)
            audit["identity_error"].append(float(page.identity_error().max()))
            audit["final_norm_distinct"].append(
                not torch.allclose(page.final_norm, page.state[-1], atol=1e-6)
            )
            pages.append(page)
    summary = {
        "n_pages": len(pages),
        "max_identity_error": max(audit["identity_error"], default=0.0),
        "max_module_disagreement_mixer": max(audit["module_agreement_mixer"], default=0.0),
        "max_module_disagreement_ffn": max(audit["module_agreement_ffn"], default=0.0),
        "final_norm_distinct_from_raw_boundary": all(audit["final_norm_distinct"]),
        "note": (
            "module disagreement > 0 means the hooked module output is not exactly what the "
            "layer adds to the residual; the residual-difference update is still exact"
        ),
    }
    return pages, summary


def _relative(estimate: Tensor, reference: Tensor) -> float:
    return float((estimate - reference).norm() / reference.norm().clamp_min(1e-12))


def make_stage_forward(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    boundary_layer: int,
    response: Callable[[Tensor], Tensor],
) -> Callable[[Tensor], Tensor]:
    """`S_{b_g}`를 leaf로 바꿔 끼우고 native response까지 흘리는 함수를 돌려줍니다.

    `boundary_layer == L`이면 final norm 입력을 바꿉니다. target model weight는 학습하지
    않습니다 (호출자가 requires_grad=False로 둡니다). offline sensitivity 전용입니다.
    """
    backbone = locate_backbone(model)
    n_layers = len(backbone.layers)
    target = backbone.final_norm if boundary_layer == n_layers else backbone.layers[boundary_layer]

    def forward(state_leaf: Tensor) -> Tensor:
        captured: dict[str, Tensor] = {}

        def replace(_module, args, kwargs):  # noqa: ANN001
            if kwargs and "hidden_states" in kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = state_leaf.to(kwargs["hidden_states"].dtype)
                return args, kwargs
            return (state_leaf.to(args[0].dtype), *args[1:]), kwargs

        def grab(_module, _args, output):  # noqa: ANN001
            captured["normed"] = _first_tensor(output)

        handles = [
            target.register_forward_pre_hook(replace, with_kwargs=True),
            backbone.final_norm.register_forward_hook(grab),
        ]
        try:
            backbone.module(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return response(captured["normed"].float())

    return forward


def load_frozen_model(
    model_id: str,
    *,
    revision: str | None,
    dtype: str = "bfloat16",
    device: str = "cuda",
    trust_remote_code: bool = False,
    local_files_only: bool = True,
):
    """server/submission용 frozen model load. 원격 code 실행과 다운로드를 기본으로 막습니다."""
    if trust_remote_code:
        raise SchemaIncompatible(
            "trust_remote_code=True is not allowed; add a reviewed in-repo adapter instead"
        )
    import importlib

    from .adapters import SERVER_PENDING, AdapterUnavailable

    try:
        tf = importlib.import_module("transformers")
    except ImportError as exc:
        raise AdapterUnavailable(f"{SERVER_PENDING}: transformers is not installed") from exc
    torch_dtype = getattr(torch, dtype)
    kwargs = {"revision": revision, "local_files_only": local_files_only,
              "trust_remote_code": False}
    tokenizer = tf.AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = tf.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def loaded_revision(model: nn.Module) -> str | None:
    """HF가 기록한 실제 commit hash. 없으면 None (pin 불가)."""
    config = getattr(model, "config", None)
    return getattr(config, "_commit_hash", None)
