"""Behavior label: outcome counts, sampling estimate, bounds, loss routing.

핵심 규칙입니다.

- outcome은 C / W / X / U_score / infra_error / not_started로 분리해서 셉니다.
  cap-hit(X)과 채점 불가(U_score)를 W로 합치지 않습니다.
- 계획된 trial이 모두 판정 가능할 때만 p_hat = C/N을 씁니다. 이는 sampling estimate이며
  정확한 population probability가 아닙니다.
- 미확정이 남으면 lower = C/N, upper = (C + unknown)/N을 저장합니다. 이 구간을 95% CI라고
  부르지 않습니다. midpoint나 0으로 바꿔 supervised target을 만들지 않습니다.
- label이 없으면 None으로 둡니다. 0이나 False로 바꾸지 않습니다. label 0은 실제 label이며
  missing이 아닙니다.
- original-panel label을 모든 variant의 pair label로 복사하지 않습니다.
- robust binary label은 출처와 frozen definition이 있을 때만 만듭니다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# slot outcome 코드. cap-hit은 X이며 W로 합치지 않습니다.
OUTCOME_CORRECT = "C"
OUTCOME_WRONG = "W"
OUTCOME_CAP_HIT = "X"
OUTCOME_UNSCORED = "U_score"
OUTCOME_INFRA_ERROR = "infra_error"
OUTCOME_NOT_STARTED = "not_started"
OUTCOMES = (
    OUTCOME_CORRECT,
    OUTCOME_WRONG,
    OUTCOME_CAP_HIT,
    OUTCOME_UNSCORED,
    OUTCOME_INFRA_ERROR,
    OUTCOME_NOT_STARTED,
)
# 판정된 outcome은 C와 W뿐입니다. 나머지는 모두 unresolved입니다.
RESOLVED_OUTCOMES = (OUTCOME_CORRECT, OUTCOME_WRONG)
UNRESOLVED_OUTCOMES = (
    OUTCOME_CAP_HIT,
    OUTCOME_UNSCORED,
    OUTCOME_INFRA_ERROR,
    OUTCOME_NOT_STARTED,
)

SEMANTIC_VERIFIED = "verified"
SEMANTIC_REJECTED = "rejected"
SEMANTIC_UNKNOWN = "unknown"
SEMANTIC_STATES = (SEMANTIC_VERIFIED, SEMANTIC_REJECTED, SEMANTIC_UNKNOWN)

# pair drop 단위. accuracy 차이이므로 [-1, 1] 범위입니다.
DROP_UNIT = "accuracy_difference_over_planned_trials"


def policy_hash(payload: dict) -> str:
    """model/thinking/sampling/scorer/budget 설정 묶음의 짧은 hash."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class PromptOutcome:
    """한 prompt(original 또는 variant)에 대한 반복 실행 결과.

    같은 original의 실행 결과는 여러 variants에서 참조할 수 있지만, 독립 관측으로
    중복 집계하지 않습니다 (prompt_id가 같으면 같은 관측입니다).
    """

    prompt_id: str
    counts: dict[str, int]
    planned_trials: int
    completed_trials: int
    termination_reason: str = "completed"
    policy_hash: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.counts) - set(OUTCOMES)
        if unknown:
            raise ValueError(f"unknown outcome code(s): {sorted(unknown)}")
        for code in OUTCOMES:
            self.counts.setdefault(code, 0)
        if self.planned_trials < 0 or self.completed_trials < 0:
            raise ValueError("trial counts must be non-negative")

    @property
    def n_correct(self) -> int:
        return int(self.counts[OUTCOME_CORRECT])

    @property
    def n_resolved(self) -> int:
        return sum(int(self.counts[code]) for code in RESOLVED_OUTCOMES)

    @property
    def n_unresolved(self) -> int:
        return sum(int(self.counts[code]) for code in UNRESOLVED_OUTCOMES)

    @property
    def fully_resolved(self) -> bool:
        """계획된 trial이 모두 실행되고 전부 C/W로 판정된 경우에만 True."""
        return (
            self.planned_trials > 0
            and self.completed_trials == self.planned_trials
            and self.n_resolved == self.planned_trials
            and self.n_unresolved == 0
        )

    def p_hat(self) -> float | None:
        """정답률 sampling estimate. 미확정이 남으면 None입니다."""
        if not self.fully_resolved:
            return None
        return self.n_correct / self.planned_trials

    def bounds(self) -> tuple[float, float]:
        """(lower, upper) = (C/N, (C + unknown)/N). 신뢰구간이 아닙니다."""
        n = max(self.planned_trials, 1)
        lower = self.n_correct / n
        upper = (self.n_correct + self.n_unresolved) / n
        return lower, min(upper, 1.0)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> PromptOutcome:
        return cls(
            prompt_id=payload["prompt_id"],
            counts=dict(payload.get("counts", {})),
            planned_trials=int(payload["planned_trials"]),
            completed_trials=int(payload["completed_trials"]),
            termination_reason=payload.get("termination_reason", "completed"),
            policy_hash=payload.get("policy_hash", ""),
        )


@dataclass
class PairLabel:
    """original-variant pair의 behavior label. 없는 값은 None으로 둡니다."""

    original_id: str
    variant_id: str
    panel_id: str
    signed_drop: float | None = None  # p_original - p_variant, [-1, 1]
    drop_lower: float | None = None
    drop_upper: float | None = None
    n_original: int = 0
    n_variant: int = 0
    resolved: bool = False  # 양쪽 prompt가 모두 fully_resolved인지
    semantic_valid: str = SEMANTIC_UNKNOWN
    unit: str = DROP_UNIT
    policy_hash: str = ""
    label_source: str | None = None
    label_version: str | None = None
    exclusion_reason: str | None = None

    @property
    def has_drop(self) -> bool:
        """behavior loss에 쓸 수 있는 실제 drop label이 있는지."""
        return self.signed_drop is not None

    @property
    def usable_for_flow(self) -> bool:
        """Page/semantic validity만 유효하면 flow에는 쓸 수 있습니다."""
        return self.semantic_valid == SEMANTIC_VERIFIED

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PanelCoverage:
    """기대한 panel 구성원과 실제 확보된 구성원."""

    expected_members: list[str] = field(default_factory=list)
    actual_members: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.expected_members) and set(self.expected_members) <= set(
            self.actual_members
        )

    @property
    def ratio(self) -> float:
        if not self.expected_members:
            return 0.0
        covered = len(set(self.expected_members) & set(self.actual_members))
        return covered / len(self.expected_members)


@dataclass
class PanelLabel:
    """original-panel 단위 label. robust label과 max-drop을 분리해 둡니다."""

    original_id: str
    panel_id: str
    robust_label: int | None = None  # 0 또는 1. None은 missing이며 0과 다릅니다.
    robust_source: str | None = None
    robust_definition: str | None = None
    max_drop: float | None = None
    max_drop_is_panel_only: bool = False  # pair counts에서 파생되지 않은 독립 target인지
    coverage: PanelCoverage = field(default_factory=PanelCoverage)
    policy_hash: str = ""

    @property
    def has_robust(self) -> bool:
        return self.robust_label is not None

    @property
    def has_max_drop(self) -> bool:
        return self.max_drop is not None

    def as_dict(self) -> dict:
        payload = asdict(self)
        return payload


@dataclass
class BinaryRobustPolicy:
    """연구용 binary robust labeling을 켜는 frozen definition.

    enabled가 False이면 robust label은 항상 None입니다. 정답률·난이도·R1 풀이로
    robust label을 만들지 않습니다. 원본부터 못 푼 경우를 자동 non-robust로 만들지도
    않습니다.
    """

    enabled: bool = False
    definition_id: str | None = None
    source: str | None = None

    def check(self) -> None:
        if self.enabled and not (self.definition_id and self.source):
            raise ValueError(
                "binary robust labeling requires a frozen definition_id and source; "
                "leave it disabled to keep robust_label null"
            )


def pair_drop(
    original: PromptOutcome,
    variant: PromptOutcome,
    *,
    original_id: str,
    variant_id: str,
    panel_id: str,
    semantic_valid: str = SEMANTIC_UNKNOWN,
    label_source: str | None = None,
    label_version: str | None = None,
) -> PairLabel:
    """두 prompt outcome에서 signed pair drop과 bounds를 만듭니다.

    d_hat_observed = p_original_hat - p_variant_hat. 한쪽이라도 미확정이면 signed_drop은
    None으로 남기고 bounds만 저장합니다 (behavior loss에서 제외).
    """
    if semantic_valid not in SEMANTIC_STATES:
        raise ValueError(f"semantic_valid must be one of {SEMANTIC_STATES}")
    if original.policy_hash != variant.policy_hash:
        raise ValueError(
            "pair drop requires the same model/thinking/sampling/scorer policy on both "
            f"prompts: {original.policy_hash!r} vs {variant.policy_hash!r}"
        )
    o_lower, o_upper = original.bounds()
    v_lower, v_upper = variant.bounds()
    label = PairLabel(
        original_id=original_id,
        variant_id=variant_id,
        panel_id=panel_id,
        # drop 하한은 original 하한 - variant 상한입니다.
        drop_lower=o_lower - v_upper,
        drop_upper=o_upper - v_lower,
        n_original=original.planned_trials,
        n_variant=variant.planned_trials,
        semantic_valid=semantic_valid,
        policy_hash=original.policy_hash,
        label_source=label_source,
        label_version=label_version,
    )
    p_o, p_v = original.p_hat(), variant.p_hat()
    if p_o is None or p_v is None:
        label.exclusion_reason = "unresolved_trials"
        return label
    label.signed_drop = p_o - p_v
    label.resolved = True
    if semantic_valid != SEMANTIC_VERIFIED:
        # 의미 보존이 확인되지 않은 pair는 drop을 계산해도 supervision으로 쓰지 않습니다.
        label.signed_drop = None
        label.exclusion_reason = f"semantic_{semantic_valid}"
    return label


def panel_max_drop(
    labels: list[PairLabel], coverage: PanelCoverage, *, panel_only_target: float | None = None
) -> tuple[float | None, bool]:
    """(max_drop, is_panel_only)를 돌려줍니다.

    독립적인 panel-only target이 주어지면 그것을 씁니다. 그렇지 않으면 pair drop에서
    파생한 diagnostic 값을 만들되, panel coverage가 불완전하거나 판정된 drop이 없으면
    None입니다. partial panel의 관측 최대값을 full-panel target으로 쓰지 않습니다.
    """
    if panel_only_target is not None:
        return float(panel_only_target), True
    if not coverage.complete:
        return None, False
    drops = [label.signed_drop for label in labels if label.signed_drop is not None]
    if not drops or len(drops) != len(coverage.expected_members):
        return None, False
    return max(0.0, max(drops)), False


def build_panel_label(
    original_id: str,
    panel_id: str,
    labels: list[PairLabel],
    coverage: PanelCoverage,
    *,
    policy_hash_value: str = "",
    robust_policy: BinaryRobustPolicy | None = None,
    provided_robust_label: int | None = None,
    panel_only_max_drop: float | None = None,
) -> PanelLabel:
    """panel label을 만듭니다. robust label은 정의가 있을 때만 채워집니다."""
    policy = robust_policy or BinaryRobustPolicy()
    policy.check()
    robust_label = None
    if policy.enabled and provided_robust_label is not None:
        if provided_robust_label not in (0, 1):
            raise ValueError("robust_label must be 0 or 1")
        robust_label = int(provided_robust_label)
    max_drop, panel_only = panel_max_drop(
        labels, coverage, panel_only_target=panel_only_max_drop
    )
    return PanelLabel(
        original_id=original_id,
        panel_id=panel_id,
        robust_label=robust_label,
        robust_source=policy.source if robust_label is not None else None,
        robust_definition=policy.definition_id if robust_label is not None else None,
        max_drop=max_drop,
        max_drop_is_panel_only=panel_only,
        coverage=coverage,
        policy_hash=policy_hash_value,
    )


@dataclass
class LabelStore:
    """pair/panel label 모음. loss routing용 mask는 label 존재 여부에서 나옵니다."""

    pairs: dict[str, PairLabel] = field(default_factory=dict)  # variant_id -> PairLabel
    panels: dict[str, PanelLabel] = field(default_factory=dict)  # original_id -> PanelLabel
    policy_hash: str = ""
    label_version: str = "v2"

    def coverage_report(self) -> dict:
        """제외율과 coverage를 보고합니다."""
        total = len(self.pairs)
        with_drop = sum(1 for label in self.pairs.values() if label.has_drop)
        reasons: dict[str, int] = {}
        for label in self.pairs.values():
            if not label.has_drop:
                reasons[label.exclusion_reason or "unknown"] = (
                    reasons.get(label.exclusion_reason or "unknown", 0) + 1
                )
        flow_usable = sum(1 for label in self.pairs.values() if label.usable_for_flow)
        return {
            "n_pairs": total,
            "n_pairs_with_drop": with_drop,
            "pair_drop_coverage": (with_drop / total) if total else 0.0,
            "pair_exclusion_rate": (1.0 - with_drop / total) if total else 0.0,
            "pair_exclusion_reasons": reasons,
            "n_pairs_usable_for_flow": flow_usable,
            "n_panels": len(self.panels),
            "n_panels_with_robust": sum(1 for p in self.panels.values() if p.has_robust),
            "n_panels_with_max_drop": sum(1 for p in self.panels.values() if p.has_max_drop),
            "n_panels_complete_coverage": sum(
                1 for p in self.panels.values() if p.coverage.complete
            ),
            "policy_hash": self.policy_hash,
            "label_version": self.label_version,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_hash": self.policy_hash,
            "label_version": self.label_version,
            "pairs": {k: v.as_dict() for k, v in self.pairs.items()},
            "panels": {k: v.as_dict() for k, v in self.panels.items()},
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> LabelStore:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        pairs = {k: PairLabel(**v) for k, v in payload.get("pairs", {}).items()}
        panels = {}
        for key, raw in payload.get("panels", {}).items():
            coverage = PanelCoverage(**raw.pop("coverage", {}))
            panels[key] = PanelLabel(coverage=coverage, **raw)
        return cls(
            pairs=pairs,
            panels=panels,
            policy_hash=payload.get("policy_hash", ""),
            label_version=payload.get("label_version", "v2"),
        )


def check_policy_consistency(labels: LabelStore) -> None:
    """서로 다른 model/thinking/sampling policy의 label을 섞지 않도록 막습니다."""
    hashes = {label.policy_hash for label in labels.pairs.values() if label.policy_hash}
    hashes |= {panel.policy_hash for panel in labels.panels.values() if panel.policy_hash}
    if len(hashes) > 1:
        raise ValueError(
            "label store mixes different model/thinking/sampling policies: "
            f"{sorted(hashes)}; keep one policy per label store"
        )


@dataclass
class OutcomeStore:
    """prompt별 outcome 기록. 같은 prompt를 독립 관측으로 중복 집계하지 않습니다.

    merge mode:
      new_only           : 새 prompt만 추가합니다 (기존 기록은 건드리지 않습니다).
      exact_continuation : 같은 trajectory를 이어받아 미확정만 채웁니다. planned_trials가
                           같아야 하고 이미 판정된 C/W 수가 줄어들 수 없습니다.
      request_resume     : prompt 전체를 다시 요청한 **별개 관측**입니다. 기존 counts와
                           합치지 않고 통째로 교체합니다.
    """

    outcomes: dict[str, PromptOutcome] = field(default_factory=dict)
    policy_hash: str = ""

    MERGE_MODES: tuple[str, ...] = ("new_only", "exact_continuation", "request_resume")

    def merge(self, incoming: list[PromptOutcome], mode: str = "new_only") -> dict:
        if mode not in self.MERGE_MODES:
            raise ValueError(f"merge mode must be one of {self.MERGE_MODES}, got {mode!r}")
        report = {"mode": mode, "added": 0, "continued": 0, "replaced": 0, "skipped": 0}
        for record in incoming:
            if self.policy_hash and record.policy_hash != self.policy_hash:
                raise ValueError(
                    f"outcome for {record.prompt_id} uses policy {record.policy_hash!r} but the "
                    f"store holds {self.policy_hash!r}; keep one policy per store"
                )
            existing = self.outcomes.get(record.prompt_id)
            if existing is None:
                self.outcomes[record.prompt_id] = record
                self.policy_hash = self.policy_hash or record.policy_hash
                report["added"] += 1
                continue
            if mode == "new_only":
                report["skipped"] += 1
                continue
            if mode == "request_resume":
                # 별개 관측이므로 기존 counts와 섞지 않고 교체합니다.
                self.outcomes[record.prompt_id] = record
                report["replaced"] += 1
                continue
            # exact_continuation: 같은 trajectory를 이어받은 경우만 허용합니다.
            if record.planned_trials != existing.planned_trials:
                raise ValueError(
                    f"exact continuation for {record.prompt_id} changed planned_trials "
                    f"{existing.planned_trials} -> {record.planned_trials}"
                )
            if record.n_correct < existing.n_correct or (
                record.counts[OUTCOME_WRONG] < existing.counts[OUTCOME_WRONG]
            ):
                raise ValueError(
                    f"exact continuation for {record.prompt_id} lost already resolved trials; "
                    "use request_resume for an independent re-run"
                )
            self.outcomes[record.prompt_id] = record
            report["continued"] += 1
        return report

    def report(self) -> dict:
        resolved = sum(1 for record in self.outcomes.values() if record.fully_resolved)
        counts: dict[str, int] = dict.fromkeys(OUTCOMES, 0)
        for record in self.outcomes.values():
            for code in OUTCOMES:
                counts[code] += int(record.counts[code])
        return {
            "n_prompts": len(self.outcomes),
            "n_fully_resolved": resolved,
            "resolved_fraction": (resolved / len(self.outcomes)) if self.outcomes else 0.0,
            "slot_counts": counts,
            "policy_hash": self.policy_hash,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy_hash": self.policy_hash,
            "outcomes": {k: v.as_dict() for k, v in self.outcomes.items()},
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> OutcomeStore:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            outcomes={
                k: PromptOutcome.from_dict(v) for k, v in payload.get("outcomes", {}).items()
            },
            policy_hash=payload.get("policy_hash", ""),
        )

    @classmethod
    def from_jsonl(cls, path: str | Path) -> list[PromptOutcome]:
        """서버에서 만든 outcome JSONL을 읽습니다."""
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [PromptOutcome.from_dict(json.loads(line)) for line in lines if line.strip()]


def build_label_store(
    outcomes: OutcomeStore,
    pairs: list,
    robust_policy: BinaryRobustPolicy | None = None,
) -> LabelStore:
    """outcome store와 pair candidate 목록에서 LabelStore를 만듭니다.

    pairs의 각 원소는 original_id / variant_id / panel_id / semantic_valid를 가진 객체여야
    합니다 (`adapters.perturbation.PairCandidate`). C4 필터를 적용하지 않고 성공·실패·
    성능 유지·개선 사례를 모두 보존합니다.
    """
    policy = robust_policy or BinaryRobustPolicy()
    policy.check()
    store = LabelStore(policy_hash=outcomes.policy_hash)
    by_original: dict[str, list] = {}
    for candidate in pairs:
        by_original.setdefault(candidate.original_id, []).append(candidate)
    for original_id, candidates in by_original.items():
        original_outcome = outcomes.outcomes.get(original_id)
        panel_id = candidates[0].panel_id
        expected, actual, labels = [], [], []
        for candidate in candidates:
            variant_outcome = outcomes.outcomes.get(candidate.variant_id)
            if original_outcome is None or variant_outcome is None:
                store.pairs[candidate.variant_id] = PairLabel(
                    original_id=original_id,
                    variant_id=candidate.variant_id,
                    panel_id=panel_id,
                    semantic_valid=candidate.semantic_valid,
                    policy_hash=outcomes.policy_hash,
                    exclusion_reason="missing_outcome_record",
                )
                continue
            label = pair_drop(
                original_outcome,
                variant_outcome,
                original_id=original_id,
                variant_id=candidate.variant_id,
                panel_id=panel_id,
                semantic_valid=candidate.semantic_valid,
                label_source=candidate.source_dataset,
                label_version=candidate.source_revision,
            )
            store.pairs[candidate.variant_id] = label
            expected.append(candidate.variant_id)
            labels.append(label)
            if label.has_drop:
                actual.append(candidate.variant_id)
        coverage = PanelCoverage(expected_members=expected, actual_members=actual)
        store.panels[original_id] = build_panel_label(
            original_id,
            panel_id,
            labels,
            coverage,
            policy_hash_value=outcomes.policy_hash,
            robust_policy=policy,
            provided_robust_label=None,  # 제공된 label이 없으면 null로 남깁니다.
        )
    return store
