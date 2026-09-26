"""Perturbation 데이터 경로: 검증된 pair import와 아주 좁은 자동 변형.

DeepMath에는 필요한 original-variant panel이 완성돼 있다고 가정하지 않습니다. 기본 경로는
**이미 검증된 pair를 import하는 것**이고, 자동 생성은 허용 범위를 확인한 두 가지만
적용합니다.

  1) formatting / 동치 표기 (allowlist된 표기 치환만)
  2) 엄격히 scope가 검증된 alpha-renaming (변수 이름만 바꾸기)

자유형 수학 문제에 substring replace나 sentence shuffle을 무조건 적용한 뒤 의미보존이라고
표시하지 않습니다. 일반적인 수학적 동치 검증을 구현했다고 주장하지 않으며, 지원하지 않는
경우는 ``pending_verification``으로 남깁니다. 정답이 같다는 이유만으로 semantic-valid로
판정하지 않습니다.

수치·조건·연산을 본질적으로 바꾸는 변형은 meaning-preserving primary와 분리해
``stress`` namespace에 둡니다.

모델의 실패 여부를 보고 perturbation을 다시 만들지 않습니다. 후보와 split은 행동 실행
전에 freeze합니다. 이 module은 외부 유료 생성 API를 호출하지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..labels import SEMANTIC_REJECTED, SEMANTIC_UNKNOWN, SEMANTIC_VERIFIED
from . import AdapterUnavailable

NAMESPACE_MEANING_PRESERVING = "meaning_preserving"
NAMESPACE_STRESS = "stress"
NAMESPACES = (NAMESPACE_MEANING_PRESERVING, NAMESPACE_STRESS)

TRANSFORM_IMPORTED = "imported"
TRANSFORM_FORMATTING = "formatting"
TRANSFORM_ALPHA_RENAME = "alpha_rename"
TRANSFORM_STRESS = "stress_edit"

# 대회 제출 사용 가능 여부는 별도 provenance 필드로 관리합니다.
USABILITY_RESEARCH_ONLY = "research_only"
USABILITY_UNVERIFIED = "unverified"
USABILITY_COMPETITION_VERIFIED = "competition_verified"
USABILITY_STATES = (
    USABILITY_RESEARCH_ONLY,
    USABILITY_UNVERIFIED,
    USABILITY_COMPETITION_VERIFIED,
)

# 허용 범위를 확인한 표기 치환만 자동 적용합니다. 의미를 바꾸지 않는 표기 수준입니다.
FORMATTING_RULES: dict[str, tuple[str, str]] = {
    "dfrac_to_frac": (r"\\dfrac", r"\\frac"),
    "tfrac_to_frac": (r"\\tfrac", r"\\frac"),
    "drop_left_right": (r"\\left\(|\\right\)", ""),
    "collapse_spaces": (r"[ \t]{2,}", " "),
}

# alpha-rename에서 건드리면 안 되는 토큰.
RESERVED_TOKENS = frozenset({"e", "i", "pi", "d", "log", "ln", "sin", "cos", "tan", "max", "min"})
UNIT_TOKENS = frozenset(
    {"m", "s", "kg", "g", "cm", "mm", "km", "l", "ml", "h", "hr", "min", "sec", "ft", "in", "mi"}
)
# 순서/지시어 표현. rename이 이 토큰을 건드리면 거부합니다.
ORDER_TOKENS = frozenset(
    {"first", "second", "third", "next", "then", "former", "latter", "above", "below", "previous"}
)


@dataclass
class PairCandidate:
    """검증 상태를 함께 들고 다니는 original-variant 후보.

    semantic_valid가 verified가 아니면 behavior supervision에 쓰지 않습니다.
    """

    pair_id: str
    original_id: str
    variant_id: str
    panel_id: str
    original_text: str
    variant_text: str
    original_answer: str
    variant_answer: str
    # 정답 대응 방식. 같으면 "identical", 다르면 명시된 mapping 설명이 필요합니다.
    answer_mapping: str = "identical"
    transform: str = TRANSFORM_IMPORTED
    namespace: str = NAMESPACE_MEANING_PRESERVING
    semantic_valid: str = SEMANTIC_UNKNOWN
    evidence: list[str] = field(default_factory=list)
    source_dataset: str = "unknown"
    source_revision: str = "unknown"
    usability: str = USABILITY_RESEARCH_ONLY
    rejection_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def usable_for_behavior(self) -> bool:
        return (
            self.semantic_valid == SEMANTIC_VERIFIED
            and self.namespace == NAMESPACE_MEANING_PRESERVING
        )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text)


def _question_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.?!]\s+", text) if part.strip().endswith("?")] or [
        text.strip().split("\n")[-1]
    ]


def check_rename_scope(text: str, old: str, new: str) -> list[str]:
    """alpha-rename이 안전한지 검사합니다. 실패 이유 목록을 돌려줍니다 (빈 목록 = 통과)."""
    reasons: list[str] = []
    if not re.fullmatch(r"[A-Za-z]", old) or not re.fullmatch(r"[A-Za-z]", new):
        reasons.append("only_single_letter_variables_supported")
        return reasons
    tokens = set(_tokens(text))
    if old.lower() in RESERVED_TOKENS or new.lower() in RESERVED_TOKENS:
        reasons.append("reserved_math_symbol")
    if old.lower() in UNIT_TOKENS or new.lower() in UNIT_TOKENS:
        reasons.append("unit_token")
    if old.lower() in ORDER_TOKENS or new.lower() in ORDER_TOKENS:
        reasons.append("order_or_deictic_token")
    if old not in tokens:
        reasons.append("old_name_not_a_standalone_token")
    if new in tokens:
        reasons.append("new_name_collides_with_existing_token")
    # 지시어가 있는 문제는 rename 후 참조가 깨질 수 있어 자동 처리하지 않습니다.
    lowered = set(token.lower() for token in _tokens(text))
    if lowered & {"former", "latter", "above", "below"}:
        reasons.append("deictic_reference_present")
    return reasons


def apply_formatting(text: str, rules: list[str]) -> str:
    """allowlist된 표기 규칙만 적용합니다. 목록에 없는 규칙은 오류입니다."""
    out = text
    for rule in rules:
        if rule not in FORMATTING_RULES:
            raise AdapterUnavailable(
                f"formatting rule {rule!r} is not in the verified allowlist "
                f"{sorted(FORMATTING_RULES)}"
            )
        pattern, replacement = FORMATTING_RULES[rule]
        out = re.sub(pattern, replacement, out)
    return out


def apply_alpha_rename(text: str, old: str, new: str) -> tuple[str, list[str]]:
    """변수 이름만 바꿉니다. scope 검사를 통과하지 못하면 원문과 이유를 돌려줍니다."""
    reasons = check_rename_scope(text, old, new)
    if reasons:
        return text, reasons
    renamed = re.sub(rf"(?<![A-Za-z_0-9]){re.escape(old)}(?![A-Za-z_0-9])", new, text)
    before = _question_sentences(text)
    after = _question_sentences(renamed)
    if len(before) != len(after):
        return text, ["question_structure_changed"]
    for a, b in zip(before, after, strict=True):
        if a.replace(old, new) != b:
            return text, ["question_text_changed_beyond_rename"]
    return renamed, []


def make_formatting_variant(
    base: PairCandidate, rules: list[str], variant_suffix: str = "fmt"
) -> PairCandidate:
    """표기만 바꾼 variant를 만듭니다. 정답 mapping은 identical입니다."""
    variant_text = apply_formatting(base.original_text, rules)
    verified = variant_text != base.original_text
    return PairCandidate(
        pair_id=f"{base.original_id}#{variant_suffix}",
        original_id=base.original_id,
        variant_id=f"{base.original_id}#{variant_suffix}",
        panel_id=base.panel_id,
        original_text=base.original_text,
        variant_text=variant_text,
        original_answer=base.original_answer,
        variant_answer=base.original_answer,
        answer_mapping="identical",
        transform=TRANSFORM_FORMATTING,
        namespace=NAMESPACE_MEANING_PRESERVING,
        semantic_valid=SEMANTIC_VERIFIED if verified else SEMANTIC_UNKNOWN,
        evidence=[f"formatting_rules={sorted(rules)}"]
        if verified
        else ["formatting_rules_had_no_effect"],
        source_dataset=base.source_dataset,
        source_revision=base.source_revision,
        usability=USABILITY_RESEARCH_ONLY,
        rejection_reasons=[] if verified else ["no_change_applied"],
    )


def make_alpha_rename_variant(
    base: PairCandidate, old: str, new: str, variant_suffix: str | None = None
) -> PairCandidate:
    """scope가 검증된 alpha-rename variant를 만듭니다. 실패하면 unknown으로 남깁니다."""
    suffix = variant_suffix or f"ren_{old}{new}"
    variant_text, reasons = apply_alpha_rename(base.original_text, old, new)
    ok = not reasons
    return PairCandidate(
        pair_id=f"{base.original_id}#{suffix}",
        original_id=base.original_id,
        variant_id=f"{base.original_id}#{suffix}",
        panel_id=base.panel_id,
        original_text=base.original_text,
        variant_text=variant_text,
        original_answer=base.original_answer,
        variant_answer=base.original_answer,
        answer_mapping="identical",
        transform=TRANSFORM_ALPHA_RENAME,
        namespace=NAMESPACE_MEANING_PRESERVING,
        semantic_valid=SEMANTIC_VERIFIED if ok else SEMANTIC_UNKNOWN,
        evidence=[f"alpha_rename {old}->{new} passed scope checks"] if ok else [],
        source_dataset=base.source_dataset,
        source_revision=base.source_revision,
        usability=USABILITY_RESEARCH_ONLY,
        rejection_reasons=reasons,
    )


REQUIRED_IMPORT_FIELDS = (
    "original_id",
    "variant_id",
    "original_text",
    "variant_text",
    "original_answer",
    "variant_answer",
    "semantic_validation_evidence",
    "source_revision",
)


def import_verified_pairs(path: str | Path) -> tuple[list[PairCandidate], dict]:
    """검증된 pair JSONL을 읽습니다.

    semantic validation evidence가 없으면 verified로 올리지 않습니다. 정답이 같다는
    사실만으로는 semantic-valid가 되지 않습니다.
    """
    path = Path(path)
    if not path.exists():
        raise AdapterUnavailable(f"verified pair file not found: {path}")
    candidates: list[PairCandidate] = []
    report = {"n_rows": 0, "n_verified": 0, "n_unknown": 0, "n_rejected": 0, "reasons": {}}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        report["n_rows"] += 1
        missing = [name for name in REQUIRED_IMPORT_FIELDS if not raw.get(name)]
        evidence = list(raw.get("semantic_validation_evidence") or [])
        namespace = raw.get("namespace", NAMESPACE_MEANING_PRESERVING)
        if namespace not in NAMESPACES:
            raise ValueError(f"row {index}: unknown namespace {namespace!r}")
        usability = raw.get("usability", USABILITY_RESEARCH_ONLY)
        if usability not in USABILITY_STATES:
            raise ValueError(f"row {index}: unknown usability {usability!r}")
        state = SEMANTIC_UNKNOWN
        reasons = []
        if missing:
            reasons.append(f"missing_fields={missing}")
        elif raw.get("semantic_valid") == SEMANTIC_REJECTED:
            state = SEMANTIC_REJECTED
            reasons.append("marked_rejected_by_source")
        elif evidence and raw.get("semantic_valid") == SEMANTIC_VERIFIED:
            state = SEMANTIC_VERIFIED
        else:
            reasons.append("no_semantic_validation_evidence")
        for reason in reasons:
            report["reasons"][reason] = report["reasons"].get(reason, 0) + 1
        report[
            "n_verified" if state == SEMANTIC_VERIFIED
            else "n_rejected" if state == SEMANTIC_REJECTED
            else "n_unknown"
        ] += 1
        original_id = raw.get("original_id") or f"row{index}"
        candidates.append(
            PairCandidate(
                pair_id=raw.get("pair_id") or f"{original_id}#{raw.get('variant_id', index)}",
                original_id=original_id,
                variant_id=raw.get("variant_id") or f"{original_id}#var{index}",
                panel_id=raw.get("panel_id") or f"{original_id}#panel",
                original_text=str(raw.get("original_text", "")),
                variant_text=str(raw.get("variant_text", "")),
                original_answer=str(raw.get("original_answer", "")),
                variant_answer=str(raw.get("variant_answer", "")),
                answer_mapping=raw.get("answer_mapping", "identical"),
                transform=raw.get("transform", TRANSFORM_IMPORTED),
                namespace=namespace,
                semantic_valid=state,
                evidence=evidence,
                source_dataset=raw.get("source_dataset", "unknown"),
                source_revision=raw.get("source_revision", "unknown"),
                usability=usability,
                rejection_reasons=reasons,
            )
        )
    return candidates, report


@dataclass
class FrozenPairStore:
    """행동 실행 전에 freeze한 후보 목록. hash로 변경을 감지합니다."""

    candidates: list[PairCandidate]
    frozen_hash: str

    @classmethod
    def freeze(cls, candidates: list[PairCandidate]) -> FrozenPairStore:
        blob = json.dumps(
            [candidate.as_dict() for candidate in candidates], sort_keys=True, default=str
        ).encode()
        return cls(candidates=candidates, frozen_hash=hashlib.sha256(blob).hexdigest()[:16])

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        path = Path(path)
        if path.exists() and not overwrite:
            raise AdapterUnavailable(
                f"{path} already exists; frozen candidates are not silently replaced "
                "(pass overwrite=True only when intentionally re-freezing before any behavior run)"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frozen_hash": self.frozen_hash,
            "n_candidates": len(self.candidates),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FrozenPairStore:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        candidates = [PairCandidate(**raw) for raw in payload["candidates"]]
        store = cls.freeze(candidates)
        if store.frozen_hash != payload["frozen_hash"]:
            raise ValueError(
                f"frozen candidate hash mismatch: stored {payload['frozen_hash']} "
                f"vs recomputed {store.frozen_hash}"
            )
        return store

    def report(self) -> dict:
        by_state: dict[str, int] = {}
        by_namespace: dict[str, int] = {}
        by_usability: dict[str, int] = {}
        for candidate in self.candidates:
            by_state[candidate.semantic_valid] = by_state.get(candidate.semantic_valid, 0) + 1
            by_namespace[candidate.namespace] = by_namespace.get(candidate.namespace, 0) + 1
            by_usability[candidate.usability] = by_usability.get(candidate.usability, 0) + 1
        return {
            "frozen_hash": self.frozen_hash,
            "n_candidates": len(self.candidates),
            "by_semantic_valid": by_state,
            "by_namespace": by_namespace,
            "by_usability": by_usability,
            "n_usable_for_behavior": sum(
                1 for c in self.candidates if c.usable_for_behavior
            ),
            "note": (
                "semantic-valid는 제공된 evidence에 근거합니다. 일반적인 수학적 동치 검증을"
                " 구현하지 않았습니다."
            ),
        }
