"""all-token native macro 관측(NativePage)과 그 schema.

기존 Page(`aimo-page-store-v2`)는 17 landmark의 sparse 관측이고 LRT/Flow/Behavior가 쓰는
좌표입니다. 이 module은 그것과 **분리된** 새 계약입니다.

- 실제 raw hidden dimension `H`, prompt의 **모든 token**, 실제 macro boundary의 raw state,
  macro별 Mixer / FFN residual update 합을 담습니다.
- 기존 sparse Page를 이 형식으로 자동 변환하지 않습니다. 필요한 all-token state와 token map이
  없으므로 `SCHEMA_INCOMPATIBLE`입니다. 17 landmark나 projection dimension을 all-token 또는
  native `H`로 취급하지 않습니다.
- `state[G]`는 마지막 decoder layer 출력(raw final boundary)이고 `final_norm`은 final norm
  이후 hidden입니다. 둘은 서로 다른 관측 위치이며 따로 기록합니다.

좌표 (한 prompt, padding 없음):

    state     [G+1, T, H]   S_{b_g}: macro g 입구(g < G)와 마지막 출구(g = G)의 raw state
    mixer     [G, T, H]     M_g = Σ_{l in macro g} actual Mixer residual update
    ffn       [G, T, H]     F_g = Σ_{l in macro g} actual FFN residual update
    final_norm [T, H]       final norm 이후 hidden (선택)

residual identity: `S_{b_{g+1}} - S_{b_g} ≈ M_g + F_g`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

NATIVE_PAGE_SCHEMA = "aimo-native-page-v1"
NATIVE_STORE_TENSORS = "native_pages.pt"
NATIVE_STORE_INDEX = "native_index.json"

# 데이터/계약 상태. 실제 데이터가 없거나 형식이 맞지 않을 때 추측으로 채우지 않습니다.
STATUS_SCHEMA_INCOMPATIBLE = "SCHEMA_INCOMPATIBLE"
STATUS_DATA_LIMIT = "DATA_LIMIT"
STATUS_DATA_UNAVAILABLE = "DATA_UNAVAILABLE"

# 관측 위치. raw block boundary와 final norm 이후 hidden을 구분합니다.
OBS_RAW_BOUNDARY = "raw_block_boundary"
OBS_FINAL_NORM = "post_final_norm"

# token span map의 segment code.
SEG_TEMPLATE = 0  # chat template / system / generation prompt token
SEG_PROBLEM = 1  # 문제 본문 token (char span은 문제 text 기준)
SEG_PREFIX = 2  # target model이 직접 생성한 짧은 native prefix
SEG_PROBE = 3  # contrast / style continuation probe
SEGMENT_NAMES = {
    SEG_TEMPLATE: "template",
    SEG_PROBLEM: "problem",
    SEG_PREFIX: "native_prefix",
    SEG_PROBE: "probe",
}

# float32 forward 기준 residual identity 상대 오차 허용치. bf16은 더 넓게 둡니다.
IDENTITY_TOL = {"float32": 1e-4, "bfloat16": 2e-2, "float16": 1e-2}


class SchemaIncompatible(ValueError):
    """입력 artifact가 native all-token 계약을 만족하지 않습니다."""

    status = STATUS_SCHEMA_INCOMPATIBLE


class DataLimit(ValueError):
    """관측이 계약을 채울 만큼 충분하지 않습니다 (예: 무단 truncation이 필요한 길이)."""

    status = STATUS_DATA_LIMIT


@dataclass
class NativeProvenance:
    """NativePage와 Stage-E checkpoint가 공유하는 provenance.

    revision이 `None`이면 pin되지 않은 것이며, 실제 실험 artifact에서는 거부합니다.
    """

    model_id: str
    model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    n_layers: int
    hidden_size: int
    vocab_size: int
    macro_boundaries: list[int]
    layout: str
    mixer_kinds: list[str]
    dtype: str
    backend: str
    chat_template_hash: str | None
    prompt_policy: dict = field(default_factory=dict)
    observation: dict = field(
        default_factory=lambda: {"state": OBS_RAW_BOUNDARY, "final": OBS_FINAL_NORM}
    )
    padding_side: str = "right"
    dataset: dict = field(default_factory=dict)
    extractor: str = "aimo.native_extract.extract_native"

    @property
    def n_macro(self) -> int:
        return len(self.macro_boundaries) - 1

    def as_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def model_key(self) -> dict:
        """다른 Page/checkpoint와 좌표가 같은지 비교할 때 쓰는 부분."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "macro_boundaries": list(self.macro_boundaries),
            "layout": self.layout,
        }

    def validate(self) -> None:
        bounds = list(self.macro_boundaries)
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != self.n_layers:
            raise SchemaIncompatible(
                f"macro boundaries {bounds} must start at 0 and end at n_layers={self.n_layers}"
            )
        if any(right <= left for left, right in zip(bounds[:-1], bounds[1:], strict=True)):
            raise SchemaIncompatible(f"macro boundaries are not strictly increasing: {bounds}")
        if len(self.mixer_kinds) != self.n_layers:
            raise SchemaIncompatible("mixer_kinds must list one kind per decoder layer")
        if self.observation.get("state") != OBS_RAW_BOUNDARY:
            raise SchemaIncompatible("state must be observed at the raw block boundary")

    @classmethod
    def from_dict(cls, payload: dict) -> NativeProvenance:
        return cls(**payload)


@dataclass
class NativePage:
    """한 prompt의 all-token macro 관측. padding 없는 길이 T를 그대로 담습니다."""

    problem_id: str
    root_id: str
    split: str
    token_ids: Tensor  # [T] int64
    segments: Tensor  # [T] int64 (SEG_*)
    char_spans: Tensor  # [T, 2] int64, segment text 기준. 대응 문자가 없으면 -1
    state: Tensor  # [G+1, T, H]
    mixer: Tensor  # [G, T, H]
    ffn: Tensor  # [G, T, H]
    provenance: NativeProvenance
    final_norm: Tensor | None = None  # [T, H]
    text_hash: str = ""

    @property
    def n_macro(self) -> int:
        return int(self.mixer.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.state.shape[-1])

    def identity_error(self) -> Tensor:
        """macro별 상대 residual identity 오차 [G]."""
        delta = self.state[1:] - self.state[:-1]
        err = (delta - (self.mixer + self.ffn)).flatten(1).norm(dim=1)
        scale = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
        return err / scale

    def validate(self, tol: float | None = None) -> None:
        prov = self.provenance
        prov.validate()
        g, t, h = self.n_macro, self.seq_len, self.hidden_size
        expect = {
            "state": (g + 1, t, h),
            "mixer": (g, t, h),
            "ffn": (g, t, h),
            "segments": (t,),
            "char_spans": (t, 2),
        }
        for name, shape in expect.items():
            got = tuple(getattr(self, name).shape)
            if got != shape:
                raise SchemaIncompatible(f"{name} has shape {got}, expected {shape}")
        if self.final_norm is not None and tuple(self.final_norm.shape) != (t, h):
            raise SchemaIncompatible(f"final_norm has shape {tuple(self.final_norm.shape)}")
        if h != prov.hidden_size:
            raise SchemaIncompatible(
                f"page hidden size {h} differs from native H={prov.hidden_size}; a projected "
                "coordinate is not the native hidden state"
            )
        if g != prov.n_macro:
            raise SchemaIncompatible(
                f"page has {g} macro stages but provenance lists {prov.n_macro}")
        limit = IDENTITY_TOL.get(prov.dtype, 1e-3) if tol is None else tol
        worst = float(self.identity_error().max()) if g else 0.0
        if worst > limit:
            raise SchemaIncompatible(
                f"macro residual identity violated: max relative error {worst:.3e} > {limit:.1e}"
            )

    def tensors(self) -> dict[str, Tensor]:
        out = {
            "token_ids": self.token_ids,
            "segments": self.segments,
            "char_spans": self.char_spans,
            "state": self.state,
            "mixer": self.mixer,
            "ffn": self.ffn,
        }
        if self.final_norm is not None:
            out["final_norm"] = self.final_norm
        return out


def require_native(obj: object) -> NativePage:
    """legacy sparse Page를 native page로 조용히 해석하지 않습니다."""
    if isinstance(obj, NativePage):
        return obj
    from .page import Page  # 순환 import를 피하려고 지연 import합니다.

    if isinstance(obj, Page):
        raise SchemaIncompatible(
            f"{STATUS_SCHEMA_INCOMPATIBLE}: legacy Page has {obj.state.shape[1]} sparse landmarks "
            "in projected coordinates, not all tokens at native H with a token span map; it "
            "cannot be converted to a native page"
        )
    raise SchemaIncompatible(
        f"{STATUS_SCHEMA_INCOMPATIBLE}: not a NativePage: {type(obj).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_native_pages(pages: list[NativePage], directory: str | Path) -> Path:
    """tensor는 `torch.save`(tensor dict만), metadata는 JSON으로 나눠 저장합니다."""
    if not pages:
        raise ValueError("no native pages to save")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = pages[0].provenance.model_key()
    for page in pages:
        page.validate()
        if page.provenance.model_key() != key:
            raise SchemaIncompatible("all pages in one store must share the same model coordinates")
    blob = {f"{i}/{name}": tensor.contiguous() for i, page in enumerate(pages)
            for name, tensor in page.tensors().items()}
    tensor_path = directory / NATIVE_STORE_TENSORS
    torch.save(blob, tensor_path)
    index = {
        "schema": NATIVE_PAGE_SCHEMA,
        "tensor_sha256": _sha256(tensor_path),
        "pages": [
            {
                "problem_id": page.problem_id,
                "root_id": page.root_id,
                "split": page.split,
                "text_hash": page.text_hash,
                "provenance": page.provenance.as_dict(),
            }
            for page in pages
        ],
    }
    path = directory / NATIVE_STORE_INDEX
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_native_pages(directory: str | Path) -> list[NativePage]:
    """schema와 tensor hash를 확인하고 `weights_only=True`로만 읽습니다 (임의 pickle 금지)."""
    directory = Path(directory)
    index_path = directory / NATIVE_STORE_INDEX
    if not index_path.exists():
        legacy = directory / "index.json"
        if legacy.exists():
            raise SchemaIncompatible(
                f"{STATUS_SCHEMA_INCOMPATIBLE}: {directory} is a legacy sparse Page store; "
                "the native path needs an all-token native page store"
            )
        raise FileNotFoundError(f"no native page index in {directory}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != NATIVE_PAGE_SCHEMA:
        raise SchemaIncompatible(
            f"native page schema {index.get('schema')!r} != {NATIVE_PAGE_SCHEMA!r}"
        )
    tensor_path = directory / NATIVE_STORE_TENSORS
    if _sha256(tensor_path) != index["tensor_sha256"]:
        raise SchemaIncompatible("native page tensor file does not match the recorded sha256")
    blob = torch.load(tensor_path, map_location="cpu", weights_only=True)
    pages = []
    for i, meta in enumerate(index["pages"]):
        page = NativePage(
            problem_id=meta["problem_id"],
            root_id=meta["root_id"],
            split=meta["split"],
            token_ids=blob[f"{i}/token_ids"],
            segments=blob[f"{i}/segments"],
            char_spans=blob[f"{i}/char_spans"],
            state=blob[f"{i}/state"],
            mixer=blob[f"{i}/mixer"],
            ffn=blob[f"{i}/ffn"],
            final_norm=blob.get(f"{i}/final_norm"),
            provenance=NativeProvenance.from_dict(meta["provenance"]),
            text_hash=meta.get("text_hash", ""),
        )
        page.validate()
        pages.append(page)
    return pages


@dataclass
class NativeBatch:
    """가변 길이 page를 오른쪽 padding으로 모은 batch. `mask`가 실제 token을 표시합니다."""

    state: Tensor  # [B, G+1, T, H]
    mixer: Tensor  # [B, G, T, H]
    ffn: Tensor  # [B, G, T, H]
    mask: Tensor  # [B, T] bool
    segments: Tensor  # [B, T]
    final_norm: Tensor | None
    problem_ids: list[str]
    root_ids: list[str]


def pad_native(pages: list[NativePage]) -> NativeBatch:
    if not pages:
        raise ValueError("empty native batch")
    key = pages[0].provenance.model_key()
    if any(page.provenance.model_key() != key for page in pages):
        raise SchemaIncompatible("cannot batch pages from different model coordinates")
    t_max = max(page.seq_len for page in pages)
    g, h = pages[0].n_macro, pages[0].hidden_size
    n = len(pages)
    state = torch.zeros(n, g + 1, t_max, h)
    mixer = torch.zeros(n, g, t_max, h)
    ffn = torch.zeros(n, g, t_max, h)
    mask = torch.zeros(n, t_max, dtype=torch.bool)
    segments = torch.full((n, t_max), -1, dtype=torch.long)
    has_final = all(page.final_norm is not None for page in pages)
    final = torch.zeros(n, t_max, h) if has_final else None
    for i, page in enumerate(pages):
        t = page.seq_len
        state[i, :, :t] = page.state
        mixer[i, :, :t] = page.mixer
        ffn[i, :, :t] = page.ffn
        mask[i, :t] = True
        segments[i, :t] = page.segments
        if final is not None:
            final[i, :t] = page.final_norm
    return NativeBatch(
        state=state,
        mixer=mixer,
        ffn=ffn,
        mask=mask,
        segments=segments,
        final_norm=final,
        problem_ids=[page.problem_id for page in pages],
        root_ids=[page.root_id for page in pages],
    )
