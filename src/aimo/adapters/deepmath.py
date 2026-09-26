"""DeepMath-103K primary 데이터 경로 adapter.

Primary source: ``zwhe99/DeepMath-103K``
참조: https://huggingface.co/datasets/zwhe99/DeepMath-103K

이 adapter가 실제로 쓰는 field는 아래 REQUIRED_FIELDS / OPTIONAL_FIELDS뿐입니다. 확인하지
못한 API나 field는 추측해서 지원한다고 쓰지 않습니다. 전체 dataset 다운로드는 서버에서만
하고, 로컬은 작은 JSONL/parquet fixture로 schema만 검증합니다.

중요한 경계:

- ``r1_solution_1/2/3``은 대상 LLM의 행동 기록도, perturbation도 아닙니다. predictor
  입력·prompt·robust label 생성에 넣지 않습니다.
- ``topic``과 ``difficulty``는 curator/split/evaluator metadata입니다. predictor input에
  넣지 않습니다.
- DeepMath difficulty를 MATH Level 1~5와 같은 척도로 취급하지 않습니다.
- 대학원 수준이라는 이유만으로 AIMO와 가깝다고 간주하지 않습니다.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import SERVER_PENDING, AdapterUnavailable

DATASET_ID = "zwhe99/DeepMath-103K"
DATASET_URL = "https://huggingface.co/datasets/zwhe99/DeepMath-103K"

# 이 adapter가 실제로 읽는 field.
REQUIRED_FIELDS = ("question", "final_answer")
OPTIONAL_FIELDS = ("difficulty", "topic", "r1_solution_1", "r1_solution_2", "r1_solution_3")
# predictor/prompt/label에 절대 넣지 않는 field.
EXCLUDED_FROM_MODEL = ("r1_solution_1", "r1_solution_2", "r1_solution_3", "topic", "difficulty")

# 우선 고려하는 주제. 자족적이고 그림 없이 이해 가능하며 final answer 채점이 신뢰성 있는
# 범위를 먼저 봅니다. geometry는 그림 의존 문항이 많아 별도 검사를 통과한 것만 씁니다.
PREFERRED_TOPIC_KEYWORDS = (
    "algebra",
    "number theory",
    "combinatorics",
    "discrete",
    "geometry",
)

# 그림/외부 자료 의존을 의심하는 표현. 걸리면 candidate에서 제외하고 이유를 남깁니다.
FIGURE_PATTERNS = (
    r"\bas shown in (?:the )?(?:figure|diagram|picture)\b",
    r"\bin the (?:figure|diagram|picture) (?:above|below)\b",
    r"\bsee (?:the )?(?:figure|diagram|attachment)\b",
    r"\[asy\]",
    r"\\includegraphics",
    r"\btable below\b",
)
_FIGURE_RE = re.compile("|".join(FIGURE_PATTERNS), re.IGNORECASE)

# split 이름. 같은 original과 그 variants/seeds는 한 split에만 둡니다.
SPLIT_NAMES = ("calibration", "train", "validation", "held_out_original", "harder")

# 서버의 첫 후보 규모. 확보된 labeled pair 수가 아니라 후보 수입니다.
DEFAULT_CANDIDATE_ORIGINALS = 300


def probe_deepmath() -> dict:
    """datasets 설치 여부만 보고합니다. 실제 다운로드는 서버에서만 합니다."""
    try:
        module = importlib.import_module("datasets")
    except ImportError:
        return {
            "available": False,
            "status": SERVER_PENDING,
            "dataset_id": DATASET_ID,
            "reason": "the 'datasets' package is an optional server dependency",
        }
    return {
        "available": True,
        "status": SERVER_PENDING,
        "dataset_id": DATASET_ID,
        "datasets_version": getattr(module, "__version__", "unknown"),
        "note": "설치는 확인했지만 실제 snapshot 다운로드와 field 확인은 서버에서 합니다.",
    }


@dataclass
class DeepMathRow:
    """DeepMath 한 row에서 이 프로젝트가 쓰는 부분만 담습니다."""

    row_id: str
    question: str
    final_answer: str
    difficulty: float | None = None
    topic: str | None = None
    source_dataset: str = DATASET_ID
    source_revision: str = "unknown"
    # r1 풀이는 존재 여부만 기록하고 내용은 보관하지 않습니다.
    n_r1_solutions: int = 0

    def text_hash(self) -> str:
        return hashlib.sha256(self.question.strip().encode()).hexdigest()[:16]

    def as_metadata(self) -> dict:
        """curator/split/evaluator metadata. model input으로 쓰지 않습니다."""
        return {
            "row_id": self.row_id,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "difficulty": self.difficulty,
            "topic": self.topic,
            "n_r1_solutions": self.n_r1_solutions,
            "difficulty_scale": "deepmath_native",  # MATH Level 1~5와 같은 척도가 아닙니다.
            "text_hash": self.text_hash(),
        }


@dataclass
class CandidateReport:
    """후보 선별 결과. 후보 수이지 확보된 labeled pair 수가 아닙니다."""

    candidates: list[DeepMathRow] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)
    n_seen: int = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def as_dict(self) -> dict:
        return {
            "n_seen": self.n_seen,
            "n_candidates": len(self.candidates),
            "rejected": dict(self.rejected),
            "note": "후보 수입니다. 확보된 labeled pair 수가 아닙니다.",
        }


def validate_row(raw: dict, row_index: int, revision: str = "unknown") -> DeepMathRow:
    """필요한 field만 검증합니다. 없는 metadata는 추측해서 채우지 않습니다."""
    missing = [name for name in REQUIRED_FIELDS if not str(raw.get(name, "")).strip()]
    if missing:
        raise ValueError(f"row {row_index} is missing required field(s): {missing}")
    difficulty = raw.get("difficulty")
    topic = raw.get("topic")
    return DeepMathRow(
        row_id=str(raw.get("row_id", raw.get("id", row_index))),
        question=str(raw["question"]),
        final_answer=str(raw["final_answer"]),
        difficulty=float(difficulty) if difficulty is not None else None,
        topic=str(topic) if topic is not None else None,
        source_revision=revision,
        n_r1_solutions=sum(
            1 for i in (1, 2, 3) if str(raw.get(f"r1_solution_{i}", "")).strip()
        ),
    )


def load_local_rows(path: str | Path, revision: str = "local") -> list[DeepMathRow]:
    """local JSONL 또는 parquet snapshot을 읽습니다 (네트워크를 쓰지 않습니다)."""
    path = Path(path)
    if not path.exists():
        raise AdapterUnavailable(f"DeepMath local snapshot not found: {path}")
    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        raw_rows = [json.loads(line) for line in lines if line.strip()]
    elif path.suffix == ".parquet":
        try:
            pyarrow = importlib.import_module("pyarrow.parquet")
        except ImportError as exc:
            raise AdapterUnavailable(
                f"{SERVER_PENDING}: reading parquet needs pyarrow (server extra)"
            ) from exc
        raw_rows = pyarrow.read_table(path).to_pylist()
    else:
        raise AdapterUnavailable(
            f"unsupported DeepMath snapshot format {path.suffix!r}; use .jsonl or .parquet"
        )
    return [validate_row(raw, i, revision) for i, raw in enumerate(raw_rows)]


def looks_self_contained(question: str) -> tuple[bool, str | None]:
    """그림/외부 자료 없이 이해 가능한지 보수적으로 검사합니다."""
    if _FIGURE_RE.search(question):
        return False, "figure_or_external_reference"
    if len(question.strip()) < 20:
        return False, "too_short_to_verify"
    return True, None


def topic_allowed(topic: str | None, allowed: tuple[str, ...] = PREFERRED_TOPIC_KEYWORDS) -> bool:
    """topic이 우선 범위에 들어오는지. topic이 없으면 추측하지 않고 제외합니다."""
    if not topic:
        return False
    lowered = topic.lower()
    return any(keyword in lowered for keyword in allowed)


def select_candidates(
    rows: list[DeepMathRow],
    max_originals: int = DEFAULT_CANDIDATE_ORIGINALS,
    allowed_topics: tuple[str, ...] = PREFERRED_TOPIC_KEYWORDS,
    require_topic: bool = True,
) -> CandidateReport:
    """후보 original을 고릅니다. 결정적이고 정렬 순서에 의존합니다."""
    report = CandidateReport()
    seen_hashes: set[str] = set()
    for row in rows:
        report.n_seen += 1
        if len(report.candidates) >= max_originals:
            report.reject("over_candidate_cap")
            continue
        if require_topic and not topic_allowed(row.topic, allowed_topics):
            report.reject("topic_not_in_scope" if row.topic else "topic_unknown")
            continue
        ok, reason = looks_self_contained(row.question)
        if not ok:
            report.reject(reason or "not_self_contained")
            continue
        text_hash = row.text_hash()
        if text_hash in seen_hashes:
            report.reject("duplicate_question")
            continue
        seen_hashes.add(text_hash)
        report.candidates.append(row)
    return report


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def benchmark_overlap(
    rows: list[DeepMathRow], benchmark_questions: list[str], ngram: int = 12
) -> dict:
    """알려진 benchmark 문항과의 표면적 겹침을 검사합니다.

    이것은 exact/near-duplicate 표면 검사일 뿐이며 **완전한 decontamination 검증이
    아닙니다**. 통과했다고 오염이 없다고 쓰지 않습니다.
    """
    def shingles(text: str) -> set[str]:
        tokens = _normalize(text).split()
        if len(tokens) < ngram:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)}

    bench = set()
    bench_exact = set()
    for question in benchmark_questions:
        bench |= shingles(question)
        bench_exact.add(_normalize(question))
    exact_hits, ngram_hits = [], []
    for row in rows:
        normalized = _normalize(row.question)
        if normalized in bench_exact:
            exact_hits.append(row.row_id)
        elif shingles(row.question) & bench:
            ngram_hits.append(row.row_id)
    return {
        "n_rows": len(rows),
        "n_benchmark_questions": len(benchmark_questions),
        "exact_matches": exact_hits,
        "ngram_matches": ngram_hits,
        "ngram": ngram,
        "decontamination_verified": False,
        "note": (
            "표면 n-gram/exact 겹침만 검사했습니다. 완전한 decontamination은 검증하지"
            " 못했습니다."
        ),
    }


def assign_splits(
    candidates: list[DeepMathRow],
    ratios: dict[str, float] | None = None,
    seed: int = 0,
) -> dict[str, list[DeepMathRow]]:
    """original 단위로 split을 배정합니다.

    같은 original과 그 variants/seeds는 한 split에만 들어갑니다 (배정 단위가 original).
    harder split은 native difficulty 상위 구간에서 고릅니다. difficulty가 없는 row는
    harder로 추측해 넣지 않습니다.
    """
    weights = ratios or {
        "calibration": 0.05,
        "train": 0.6,
        "validation": 0.1,
        "held_out_original": 0.15,
        "harder": 0.1,
    }
    unknown = set(weights) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"unknown split name(s): {sorted(unknown)}")
    # difficulty가 있는 후보 중 상위 구간을 harder로 먼저 뺍니다.
    with_difficulty = [row for row in candidates if row.difficulty is not None]
    n_harder = int(round(len(candidates) * weights.get("harder", 0.0)))
    with_difficulty.sort(key=lambda row: (-(row.difficulty or 0.0), row.row_id))
    harder = with_difficulty[:n_harder]
    harder_ids = {row.row_id for row in harder}
    rest = [row for row in candidates if row.row_id not in harder_ids]
    # 남은 후보는 row_id hash로 결정적으로 나눕니다.
    rest.sort(key=lambda row: hashlib.sha256(f"{seed}:{row.row_id}".encode()).hexdigest())
    out: dict[str, list[DeepMathRow]] = {name: [] for name in SPLIT_NAMES}
    out["harder"] = harder
    order = [name for name in SPLIT_NAMES if name != "harder"]
    total = sum(weights.get(name, 0.0) for name in order) or 1.0
    cursor = 0
    for index, name in enumerate(order):
        share = weights.get(name, 0.0) / total
        count = len(rest) - cursor if index == len(order) - 1 else int(round(len(rest) * share))
        out[name] = rest[cursor : cursor + count]
        cursor += count
    return out
