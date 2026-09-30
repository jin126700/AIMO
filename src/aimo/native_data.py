"""Stage-E discovery 데이터 adapter와 split.

discovery는 label-free입니다. teacher는 target model의 native next-token distribution이며
behavior label, correctness, robustness, max_drop, 제공 solution을 입력·teacher·loss에 쓰지
않습니다.

- DeepMath: `question`만 model 입력으로 씁니다. `topic` / `difficulty`는 split·분석
  metadata입니다. `final_answer`와 `r1_solution_*`는 읽는 순간 버리고 record에 남기지
  않습니다.
- GSM8K main/train: feasibility와 쉬운 계산 대조군입니다. `answer`(풀이 포함)는 버립니다.
- GSM-Symbolic / GSM-Plus는 discovery fitting에 섞지 않습니다.
- locked / known-test / evaluation-only ID와 그 text hash는 discovery에 들어오지 못합니다.
- 같은 original / root / template의 모든 variant와 model / effort 행은 같은 split입니다.
- numeric variant나 P1/P2를 자동으로 MP(meaning-preserving)라고 분류하지 않습니다.
- 공식 train-main-v2 label은 공간과 feature를 freeze한 뒤 작은 predictor를 학습할 때만
  씁니다. null label은 채우지 않고 제외하며 그 수를 보고합니다. label 수를 hardcode하지
  않고 실제 source와 root lineage에서 셉니다.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .native_page import STATUS_DATA_UNAVAILABLE

# discovery record / teacher / loss에 절대 들어가면 안 되는 key.
FORBIDDEN_DISCOVERY_KEYS = frozenset(
    {
        "final_answer",
        "answer",
        "solution",
        "r1_solution_1",
        "r1_solution_2",
        "r1_solution_3",
        "model_is_robust",
        "is_robust",
        "robust",
        "correct",
        "correctness",
        "max_drop",
        "pair_drop",
        "drop",
        "label",
        "labels",
        "outcome",
    }
)

# discovery fitting에 섞지 않는 source.
DENIED_DISCOVERY_SOURCES = ("gsm-symbolic", "gsm_symbolic", "gsmsymbolic", "gsm-plus",
                            "gsm_plus", "gsmplus")

SOURCE_DEEPMATH = "zwhe99/DeepMath-103K"
SOURCE_GSM8K = "openai/gsm8k:main"

ROLE_DISCOVERY = "discovery"
ROLE_FEASIBILITY = "feasibility_control"

MP_VERIFIED = "MP_VERIFIED_FORMAT_VIEW"
MP_UNCLASSIFIED = "NOT_MP_UNVERIFIED"

DISCOVERY_SPLITS = ("train", "dev", "held_out")


class LabelLeak(ValueError):
    """label-free 경로에 label / answer / solution field가 들어왔습니다."""


class DataUnavailable(FileNotFoundError):
    status = STATUS_DATA_UNAVAILABLE


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()[:16]


def assert_label_free(record: dict) -> None:
    leaked = sorted(FORBIDDEN_DISCOVERY_KEYS & set(record))
    meta = record.get("meta") or {}
    leaked += sorted(f"meta.{key}" for key in FORBIDDEN_DISCOVERY_KEYS & set(meta))
    if leaked:
        raise LabelLeak(f"label-free discovery record carries forbidden field(s): {leaked}")


@dataclass
class ProtectedRegistry:
    """locked / known-test / evaluation-only 문제. ID와 정규화 text hash 둘 다로 막습니다."""

    ids: set[str] = field(default_factory=set)
    text_hashes: set[str] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)

    def add_problem(self, problem_id: str | None, text: str | None) -> None:
        if problem_id:
            self.ids.add(str(problem_id))
        if text:
            self.text_hashes.add(text_hash(text))

    def is_protected(self, problem_id: str | None, text: str | None) -> bool:
        return (problem_id is not None and str(problem_id) in self.ids) or (
            text is not None and text_hash(text) in self.text_hashes
        )

    @classmethod
    def load(cls, path: str | Path) -> ProtectedRegistry:
        """`{"ids": [...], "text_hashes": [...], "texts": [...]}` JSON 또는 공식 cases.jsonl."""
        path = Path(path)
        if not path.exists():
            raise DataUnavailable(f"{STATUS_DATA_UNAVAILABLE}: protected registry {path} not found")
        registry = cls(sources=[str(path)])
        if path.suffix == ".jsonl":
            for row in _read_jsonl(path):
                text = row.get("problem") or row.get("original_problem")
                registry.add_problem(row.get("id"), text)
            return registry
        payload = json.loads(path.read_text(encoding="utf-8"))
        registry.ids.update(str(x) for x in payload.get("ids", []))
        registry.text_hashes.update(payload.get("text_hashes", []))
        for text in payload.get("texts", []):
            registry.add_problem(None, text)
        return registry


@dataclass
class DiscoveryProblem:
    """label-free discovery 문제 하나. `text`만 model 입력입니다."""

    problem_id: str
    root_id: str
    source: str
    role: str
    text: str
    split: str = ""
    meta: dict = field(default_factory=dict)  # topic / difficulty 등 split·분석 metadata

    def record(self) -> dict:
        out = {
            "problem_id": self.problem_id,
            "root_id": self.root_id,
            "source": self.source,
            "role": self.role,
            "split": self.split,
            "text": self.text,
            "meta": dict(self.meta),
        }
        assert_label_free(out)
        return out


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_rows(path: str | Path) -> list[dict]:
    """로컬 JSONL / JSON / parquet만 읽습니다. 원격 다운로드를 하지 않습니다."""
    path = Path(path)
    if not path.exists():
        raise DataUnavailable(f"{STATUS_DATA_UNAVAILABLE}: {path} does not exist locally")
    if path.suffix == ".jsonl":
        return list(_read_jsonl(path))
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload.get("rows", [])
    if path.suffix == ".parquet":
        import pandas  # optional; 없으면 ImportError를 그대로 올립니다.

        return json.loads(pandas.read_parquet(path).to_json(orient="records"))
    raise ValueError(f"unsupported data file type: {path.suffix}")


def _check_source_allowed(source: str) -> None:
    lowered = source.lower().replace(" ", "")
    if any(denied in lowered for denied in DENIED_DISCOVERY_SOURCES):
        raise ValueError(f"{source!r} must not be mixed into discovery fitting")


def load_deepmath_discovery(
    path: str | Path,
    *,
    protected: ProtectedRegistry,
    limit: int | None = None,
    seed: int = 0,
    revision: str = "unknown",
) -> tuple[list[DiscoveryProblem], dict]:
    """새 DeepMath discovery subset. answer / solution field는 읽자마자 버립니다."""
    rows = read_rows(path)
    order = list(range(len(rows)))
    random.Random(f"deepmath-discovery:{seed}").shuffle(order)
    problems: list[DiscoveryProblem] = []
    report = Counter()
    for index in order:
        raw = rows[index]
        question = str(raw.get("question") or "").strip()
        row_id = str(raw.get("row_id", raw.get("id", index)))
        if not question:
            report["missing_question"] += 1
            continue
        if protected.is_protected(row_id, question):
            report["protected"] += 1
            continue
        root = raw.get("root_id") or raw.get("template_id") or raw.get("original_id") or row_id
        meta = {"topic": raw.get("topic"), "difficulty": raw.get("difficulty"),
                "source_revision": revision}
        problems.append(
            DiscoveryProblem(
                problem_id=f"deepmath:{row_id}",
                root_id=f"deepmath:{root}",
                source=SOURCE_DEEPMATH,
                role=ROLE_DISCOVERY,
                text=question,
                meta=meta,
            )
        )
        report["kept"] += 1
        if limit is not None and len(problems) >= limit:
            break
    return problems, dict(report)


def load_gsm8k_control(
    path: str | Path, *, protected: ProtectedRegistry, source: str = SOURCE_GSM8K
) -> tuple[list[DiscoveryProblem], dict]:
    """GSM8K main/train을 feasibility 대조군으로 읽습니다. `answer`는 버립니다."""
    _check_source_allowed(source)
    problems, report = [], Counter()
    for index, raw in enumerate(read_rows(path)):
        question = str(raw.get("question") or "").strip()
        row_id = str(raw.get("id", index))
        if not question:
            report["missing_question"] += 1
            continue
        if protected.is_protected(row_id, question):
            report["protected"] += 1
            continue
        problems.append(
            DiscoveryProblem(
                problem_id=f"gsm8k:{row_id}",
                root_id=f"gsm8k:{row_id}",
                source=source,
                role=ROLE_FEASIBILITY,
                text=question,
            )
        )
        report["kept"] += 1
    return problems, dict(report)


def assign_splits(
    problems: list[DiscoveryProblem],
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> list[DiscoveryProblem]:
    """root 단위 deterministic hash split. 같은 root의 모든 행이 같은 split에 갑니다."""
    fractions = fractions or {"train": 0.7, "dev": 0.15, "held_out": 0.15}
    names = list(fractions)
    edges, total = [], 0.0
    for name in names:
        total += fractions[name]
        edges.append(total)
    for problem in problems:
        _check_source_allowed(problem.source)
        digest = hashlib.sha256(f"{seed}:{problem.root_id}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64 * total
        problem.split = next(name for name, edge in zip(names, edges, strict=True) if u < edge)
    validate_lineage(problems)
    return problems


def validate_lineage(rows: Iterable) -> None:
    """같은 root가 여러 split에 걸쳐 있으면 실패합니다."""
    seen: dict[str, str] = {}
    for row in rows:
        root = row.root_id if hasattr(row, "root_id") else row["root_id"]
        split = row.split if hasattr(row, "split") else row["split"]
        if seen.setdefault(root, split) != split:
            raise ValueError(f"root {root!r} appears in splits {seen[root]!r} and {split!r}")


def generalization_split(
    problems: list[DiscoveryProblem], axis: str, held_values: set
) -> dict[str, list[DiscoveryProblem]]:
    """새 topic / difficulty / MP family로 frozen encoder의 일반화를 보는 split.

    held value를 가진 root 전체를 held-out으로 보내므로 lineage를 깨지 않습니다.
    """
    if axis not in ("topic", "difficulty", "mp_family"):
        raise ValueError(f"unknown generalization axis {axis!r}")
    held_roots = {p.root_id for p in problems if p.meta.get(axis) in held_values}
    return {
        "fit": [p for p in problems if p.root_id not in held_roots],
        "held_out": [p for p in problems if p.root_id in held_roots],
    }


def classify_variant(record: dict) -> str:
    """variant가 MP인지 판정합니다. 검증된 formatting view만 MP로 인정합니다.

    numeric variant, P1/P2 같은 perturbation 이름, 외부 paraphrase는 자동으로 MP가 아닙니다.
    """
    if record.get("mp_view_kind") and record.get("signature_preserved") is True:
        return MP_VERIFIED
    return MP_UNCLASSIFIED


@dataclass
class OfficialLabelRow:
    model_id: str
    dataset_id: str
    problem_id: str
    problem: str
    reasoning_effort: str
    is_robust: bool | None
    root_id: str = ""
    split: str = ""


def load_official_labels(path: str | Path) -> tuple[list[OfficialLabelRow], dict]:
    """공식 train-main-v2 / val-sample 형식 row를 읽습니다. null label은 None으로 둡니다."""
    rows: list[OfficialLabelRow] = []
    for raw in read_rows(path):
        label = raw.get("model_is_robust")
        if label is not None and type(label) is not bool:
            raise ValueError(f"model_is_robust must be bool or null, got {label!r}")
        rows.append(
            OfficialLabelRow(
                model_id=str(raw["model_id"]),
                dataset_id=str(raw.get("dataset_id", "")),
                problem_id=str(raw.get("problem_id", "")),
                problem=str(raw["original_problem"]),
                reasoning_effort=str(raw.get("reasoning_effort") or "default"),
                is_robust=label,
            )
        )
    assign_official_lineage(rows)
    return rows, official_label_summary(rows)


def assign_official_lineage(rows: list[OfficialLabelRow]) -> None:
    """(dataset_id, problem_id)와 정규화 text hash를 union-find로 묶어 root를 정합니다."""
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for row in rows:
        keys = [f"text:{text_hash(row.problem)}"]
        if row.dataset_id and row.problem_id:
            keys.append(f"id:{row.dataset_id}:{row.problem_id}")
        for key in keys[1:]:
            union(keys[0], key)
        find(keys[0])
    for row in rows:
        row.root_id = "root:" + hashlib.sha256(
            find(f"text:{text_hash(row.problem)}").encode()
        ).hexdigest()[:12]


def official_label_summary(rows: list[OfficialLabelRow]) -> dict:
    by_group: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        key = f"{row.model_id}|{row.reasoning_effort}"
        state = "null" if row.is_robust is None else ("robust" if row.is_robust else "not_robust")
        by_group[key][state] += 1
    labeled = [row for row in rows if row.is_robust is not None]
    return {
        "n_rows": len(rows),
        "n_labeled": len(labeled),
        "n_null_label": len(rows) - len(labeled),
        "n_unique_roots": len({row.root_id for row in rows}),
        "n_unique_labeled_roots": len({row.root_id for row in labeled}),
        "n_models": len({row.model_id for row in rows}),
        "by_model_effort": {key: dict(value) for key, value in sorted(by_group.items())},
        "note": "counts come from the actual file; null labels are excluded, never filled",
    }


@dataclass(frozen=True)
class NativePrefix:
    """target model이 직접 생성한 짧은 prefix. dataset 제공 solution은 여기에 올 수 없습니다."""

    problem_id: str
    text: str
    generated_by: str
    generation_policy_hash: str
    source: str = "target_model_native"


def validate_native_prefix(
    prefix: NativePrefix, model_id: str, forbidden_texts: Iterable[str] = (), max_chars: int = 400
) -> NativePrefix:
    if prefix.source != "target_model_native":
        raise LabelLeak(f"native prefix source must be the target model, got {prefix.source!r}")
    if prefix.generated_by != model_id:
        raise LabelLeak(f"prefix was generated by {prefix.generated_by!r}, not {model_id!r}")
    if len(prefix.text) > max_chars:
        raise ValueError("native prefix must be short")
    norm = normalize_text(prefix.text)
    for text in forbidden_texts:
        other = normalize_text(text)
        if norm and other and (norm in other or other in norm):
            raise LabelLeak("native prefix overlaps a dataset-provided solution")
    return prefix


@dataclass(frozen=True)
class ContrastProbe:
    """짧은 continuation 쌍. 정답 annotation이나 수학 relation label이 아닙니다."""

    family: str
    text_a: str
    text_b: str
    note: str = "contrast probe; not an answer annotation or a math relation label"


CONTRAST_FAMILIES = ("number", "sign", "operator", "condition", "style", "format")


def make_contrast_probes(seed: int = 0, per_family: int = 2) -> list[ContrastProbe]:
    """숫자·부호·연산자·조건 contrast와 style / format control을 고정 seed로 만듭니다."""
    rng = random.Random(f"contrast:{seed}")
    probes = []
    for _ in range(per_family):
        a, b = rng.randint(2, 9), rng.randint(11, 29)
        probes += [
            ContrastProbe("number", f" so x = {a}", f" so x = {a + 1}"),
            ContrastProbe("sign", f" so x = {a}", f" so x = -{a}"),
            ContrastProbe("operator", f" then {a} + {b}", f" then {a} - {b}"),
            ContrastProbe("condition", f" for all n > {a}", f" for all n < {a}"),
            ContrastProbe("style", " Let us compute.", " We now compute."),
            ContrastProbe("format", f" Answer: {a}", f" Final answer: {a}"),
        ]
    return probes
