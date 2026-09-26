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

    정의와 제약:
      - `counts`는 음이 아닌 정수이고 **합이 정확히 `planned_trials`** 와 같아야 합니다.
        기록되지 않은 planned slot이 있으면 `not_started`로 명시해야 합니다.
      - `completed_trials`는 **generation이 끝난 slot 수**입니다 (채점 여부와 무관).
      - `n_resolved`는 **점수가 확정된 slot 수**(C + W)입니다. 둘은 다릅니다: cap-hit이나
        채점 불가는 generation이 끝났어도 score가 확정되지 않습니다.
      - `not_started` slot은 completed로 세지 않습니다.
      - 같은 original의 실행 결과는 여러 variants에서 참조할 수 있지만 `prompt_id`가 같으면
        같은 관측이므로 중복 집계하지 않습니다.
      - `scorer_version`이 다른 기록을 같은 store에 섞지 않습니다.
    """

    prompt_id: str
    counts: dict[str, int]
    planned_trials: int
    completed_trials: int
    termination_reason: str = "completed"
    policy_hash: str = ""
    scorer_version: str = ""
    # 같은 trajectory 이어받기를 증명하는 slot 단위 증거. 없으면 빈 dict입니다.
    slot_evidence: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.counts) - set(OUTCOMES)
        if unknown:
            raise ValueError(f"unknown outcome code(s): {sorted(unknown)}")
        for code, value in self.counts.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"count for {code!r} must be an int, got {value!r}")
            if value < 0:
                raise ValueError(f"count for {code!r} must be non-negative, got {value}")
        for code in OUTCOMES:
            self.counts.setdefault(code, 0)
        if self.planned_trials < 0 or self.completed_trials < 0:
            raise ValueError("trial counts must be non-negative")
        total = sum(self.counts[code] for code in OUTCOMES)
        if total != self.planned_trials:
            raise ValueError(
                f"outcome {self.prompt_id!r}: sum(counts)={total} must equal "
                f"planned_trials={self.planned_trials}; record unrecorded slots explicitly as "
                f"'{OUTCOME_NOT_STARTED}' instead of leaving them out"
            )
        if self.completed_trials > self.planned_trials:
            raise ValueError(
                f"outcome {self.prompt_id!r}: completed_trials={self.completed_trials} exceeds "
                f"planned_trials={self.planned_trials}"
            )
        if self.completed_trials + self.counts[OUTCOME_NOT_STARTED] > self.planned_trials:
            raise ValueError(
                f"outcome {self.prompt_id!r}: completed_trials + not_started exceeds "
                "planned_trials; not_started slots are not completed"
            )

    @property
    def n_correct(self) -> int:
        return int(self.counts[OUTCOME_CORRECT])

    @property
    def n_resolved(self) -> int:
        """점수가 확정된 slot 수 (C + W). completed_trials와 다릅니다."""
        return sum(int(self.counts[code]) for code in RESOLVED_OUTCOMES)

    @property
    def n_unresolved(self) -> int:
        return sum(int(self.counts[code]) for code in UNRESOLVED_OUTCOMES)

    @property
    def n_not_started(self) -> int:
        return int(self.counts[OUTCOME_NOT_STARTED])

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
        """정답률 sampling estimate. 미확정이 남거나 N=0이면 None입니다."""
        if not self.fully_resolved:
            return None
        return self.n_correct / self.planned_trials

    def bounds(self) -> tuple[float, float] | None:
        """(lower, upper) = (C/N, (C + unresolved)/N). N=0이면 undefined(None)입니다.

        미확정 slot은 정답일 수도 오답일 수도 있으므로 upper에 모두 포함됩니다. 따라서
        planned=4, C=1, 나머지 3이 미확정이면 bounds는 [0.25, 1.0]입니다. 이 구간은
        신뢰구간이 아닙니다.
        """
        if self.planned_trials <= 0:
            return None
        n = self.planned_trials
        lower = self.n_correct / n
        upper = (self.n_correct + self.n_unresolved) / n
        return lower, min(upper, 1.0)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: dict, *, fill_missing_as_not_started: bool = False
    ) -> PromptOutcome:
        """저장된 기록을 복원합니다.

        `fill_missing_as_not_started=True`는 **명시적인 import 규칙**입니다. 기록되지 않은
        planned slot을 `not_started`로 채웁니다. 기본값은 False이며, 합이 맞지 않는 입력은
        불완전 입력으로 거부합니다.
        """
        counts = {k: int(v) for k, v in dict(payload.get("counts", {})).items()}
        planned = int(payload["planned_trials"])
        if fill_missing_as_not_started:
            recorded = sum(counts.get(code, 0) for code in OUTCOMES)
            if recorded < planned:
                counts[OUTCOME_NOT_STARTED] = counts.get(OUTCOME_NOT_STARTED, 0) + (
                    planned - recorded
                )
        return cls(
            prompt_id=payload["prompt_id"],
            counts=counts,
            planned_trials=planned,
            completed_trials=int(payload["completed_trials"]),
            termination_reason=payload.get("termination_reason", "completed"),
            policy_hash=payload.get("policy_hash", ""),
            scorer_version=payload.get("scorer_version", ""),
            slot_evidence=dict(payload.get("slot_evidence", {})),
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
    scorer_version: str = ""
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
    """기대한 panel 구성원과 단계별 확보 상황.

    세 단계를 구분합니다.
      page_members    : Page가 존재하는 variant
      outcome_members : 행동 측정 기록이 존재하는 variant
      actual_members  : 점수가 확정되어 signed drop label이 생긴 variant

    `expected_members`는 frozen candidate manifest 전체에서 만들며, 측정 기록이 없는
    variant도 기대 구성원으로 남깁니다. `complete`는 **점수 확정 기준**입니다.
    """

    expected_members: list[str] = field(default_factory=list)
    actual_members: list[str] = field(default_factory=list)
    page_members: list[str] = field(default_factory=list)
    outcome_members: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.expected_members) and set(self.expected_members) <= set(
            self.actual_members
        )

    @property
    def ratio(self) -> float:
        return self._ratio(self.actual_members)

    @property
    def page_ratio(self) -> float:
        return self._ratio(self.page_members)

    @property
    def outcome_ratio(self) -> float:
        return self._ratio(self.outcome_members)

    @property
    def missing_members(self) -> list[str]:
        return sorted(set(self.expected_members) - set(self.actual_members))

    def _ratio(self, members: list[str]) -> float:
        if not self.expected_members:
            return 0.0
        covered = len(set(self.expected_members) & set(members))
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
    if original.scorer_version != variant.scorer_version:
        raise ValueError(
            "pair drop requires the same scorer version on both prompts: "
            f"{original.scorer_version!r} vs {variant.scorer_version!r}"
        )
    o_bounds = original.bounds()
    v_bounds = variant.bounds()
    label = PairLabel(
        original_id=original_id,
        variant_id=variant_id,
        panel_id=panel_id,
        n_original=original.planned_trials,
        n_variant=variant.planned_trials,
        semantic_valid=semantic_valid,
        policy_hash=original.policy_hash,
        scorer_version=original.scorer_version,
        label_source=label_source,
        label_version=label_version,
    )
    if o_bounds is None or v_bounds is None:
        # planned trial이 0이면 확률과 bounds가 모두 undefined입니다.
        label.exclusion_reason = "no_planned_trials"
        return label
    o_lower, o_upper = o_bounds
    v_lower, v_upper = v_bounds
    # drop 하한은 original 하한 - variant 상한입니다.
    label.drop_lower = o_lower - v_upper
    label.drop_upper = o_upper - v_lower
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
        # 누락된 variant를 제외한 maximum을 full-panel max-drop으로 쓰지 않습니다.
        return None, False
    expected = set(coverage.expected_members)
    drops = [
        label.signed_drop
        for label in labels
        if label.signed_drop is not None and label.variant_id in expected
    ]
    if not drops or len(drops) != len(expected):
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


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


@dataclass
class LabelStore:
    """pair/panel label 모음. loss routing용 mask는 label 존재 여부에서 나옵니다."""

    pairs: dict[str, PairLabel] = field(default_factory=dict)  # variant_id -> PairLabel
    panels: dict[str, PanelLabel] = field(default_factory=dict)  # original_id -> PanelLabel
    policy_hash: str = ""
    scorer_version: str = ""
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
            "mean_page_coverage": _mean(
                [p.coverage.page_ratio for p in self.panels.values()]
            ),
            "mean_outcome_coverage": _mean(
                [p.coverage.outcome_ratio for p in self.panels.values()]
            ),
            "mean_resolved_coverage": _mean([p.coverage.ratio for p in self.panels.values()]),
            "policy_hash": self.policy_hash,
            "scorer_version": self.scorer_version,
            "label_version": self.label_version,
        }

    def save(self, path: str | Path) -> Path:
        """lock을 잡고 atomic하게 저장합니다."""
        from .runtime import atomic_write_json, store_lock

        path = Path(path)
        payload = {
            "policy_hash": self.policy_hash,
            "scorer_version": self.scorer_version,
            "label_version": self.label_version,
            "pairs": {k: v.as_dict() for k, v in self.pairs.items()},
            "panels": {k: v.as_dict() for k, v in self.panels.items()},
        }
        with store_lock(path):
            atomic_write_json(path, payload)
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
            scorer_version=payload.get("scorer_version", ""),
            label_version=payload.get("label_version", "v2"),
        )



# 같은 trajectory 이어받기를 증명하는 데 필요한 slot 증거 field.
TRAJECTORY_EVIDENCE_FIELDS = ("request_id", "seed", "prompt_hash", "policy_hash",
                              "token_prefix_hash")

MERGE_NEW_ONLY = "new_only"
MERGE_EXACT_CONTINUATION = "exact_continuation"
MERGE_FILL_NOT_STARTED = "fill_not_started"
MERGE_SEPARATE_COHORT = "separate_cohort"
MERGE_MODES = (
    MERGE_NEW_ONLY,
    MERGE_EXACT_CONTINUATION,
    MERGE_FILL_NOT_STARTED,
    MERGE_SEPARATE_COHORT,
)


def _check_trajectory_evidence(existing: PromptOutcome, incoming: PromptOutcome) -> None:
    """같은 trajectory를 이어받았다는 증거를 확인합니다.

    counts 단조성만으로는 인정하지 않습니다. 기존 기록에 slot 증거가 없으면 이력을
    만들어내지 않고 거부합니다.
    """
    if not existing.slot_evidence:
        raise ValueError(
            f"{existing.prompt_id!r}: exact continuation needs per-slot trajectory evidence on "
            "the existing record; aggregate-only records cannot be continued (use "
            f"'{MERGE_SEPARATE_COHORT}' for an independent re-run)"
        )
    if not incoming.slot_evidence:
        raise ValueError(
            f"{incoming.prompt_id!r}: exact continuation needs per-slot trajectory evidence on "
            "the incoming record"
        )
    shared = set(existing.slot_evidence) & set(incoming.slot_evidence)
    if not shared:
        raise ValueError(
            f"{incoming.prompt_id!r}: incoming slots {sorted(incoming.slot_evidence)} share no "
            f"slot id with the existing record {sorted(existing.slot_evidence)}"
        )
    for slot_id in sorted(shared):
        before = existing.slot_evidence[slot_id]
        after = incoming.slot_evidence[slot_id]
        for field_name in TRAJECTORY_EVIDENCE_FIELDS:
            if field_name not in before or field_name not in after:
                raise ValueError(
                    f"{incoming.prompt_id!r} slot {slot_id!r}: missing trajectory evidence "
                    f"field {field_name!r} (required: {list(TRAJECTORY_EVIDENCE_FIELDS)})"
                )
            if before[field_name] != after[field_name]:
                raise ValueError(
                    f"{incoming.prompt_id!r} slot {slot_id!r}: {field_name} changed "
                    f"{before[field_name]!r} -> {after[field_name]!r}; this is an independent "
                    f"re-run, not an exact continuation"
                )


def _add_counts(base: dict[str, int], delta: dict[str, int]) -> dict[str, int]:
    return {code: int(base.get(code, 0)) + int(delta.get(code, 0)) for code in OUTCOMES}


@dataclass
class OutcomeStore:
    """prompt별 outcome 기록. 같은 prompt를 독립 관측으로 중복 집계하지 않습니다.

    merge mode:
      new_only           : 새 prompt만 추가합니다 (기존 기록은 건드리지 않습니다).
      exact_continuation : 같은 trajectory를 이어받아 미확정만 채웁니다. slot 단위
                           trajectory 증거가 양쪽에 있어야 하고, 이미 확정된 C/W가 줄 수
                           없습니다.
      fill_not_started   : **미시작 slot만** 실행한 delta 기록을 더합니다. 완료된 slot(C/W/X/
                           U_score/infra_error)은 건드리지 않으므로 기존 X가 새 성공으로
                           대체되지 않습니다. incoming의 planned_trials는 기존 not_started
                           수를 넘을 수 없습니다.
      separate_cohort    : 독립 재실행입니다. primary 기록을 바꾸지 않고 별도 cohort에
                           보존합니다.

    `scorer_version`이 다른 기록을 같은 store에 섞지 않습니다.
    """

    outcomes: dict[str, PromptOutcome] = field(default_factory=dict)
    policy_hash: str = ""
    scorer_version: str = ""
    cohorts: dict[str, dict[str, PromptOutcome]] = field(default_factory=dict)

    MERGE_MODES: tuple[str, ...] = MERGE_MODES

    def _check_compatible(self, record: PromptOutcome) -> None:
        if self.policy_hash and record.policy_hash != self.policy_hash:
            raise ValueError(
                f"outcome for {record.prompt_id} uses policy {record.policy_hash!r} but the "
                f"store holds {self.policy_hash!r}; keep one policy per store"
            )
        if self.scorer_version and record.scorer_version != self.scorer_version:
            raise ValueError(
                f"outcome for {record.prompt_id} was scored with version "
                f"{record.scorer_version!r} but the store holds {self.scorer_version!r}; "
                "do not overwrite earlier results with a new scorer, use a separate store"
            )

    def _adopt(self, record: PromptOutcome) -> None:
        self.policy_hash = self.policy_hash or record.policy_hash
        self.scorer_version = self.scorer_version or record.scorer_version

    def merge(
        self,
        incoming: list[PromptOutcome],
        mode: str = MERGE_NEW_ONLY,
        cohort: str | None = None,
    ) -> dict:
        """mode에 따라 기록을 반영합니다. 잘못된 입력은 조용히 통과시키지 않습니다."""
        if mode not in MERGE_MODES:
            raise ValueError(f"merge mode must be one of {MERGE_MODES}, got {mode!r}")
        if mode == MERGE_SEPARATE_COHORT and not cohort:
            raise ValueError(f"mode {MERGE_SEPARATE_COHORT!r} needs an explicit cohort name")
        seen_ids: set[str] = set()
        report = {
            "mode": mode,
            "cohort": cohort,
            "added": 0,
            "continued": 0,
            "filled": 0,
            "cohort_records": 0,
            "skipped": 0,
        }
        for record in incoming:
            if record.prompt_id in seen_ids:
                raise ValueError(
                    f"duplicate prompt_id {record.prompt_id!r} in a single merge batch"
                )
            seen_ids.add(record.prompt_id)
            self._check_compatible(record)
            if mode == MERGE_SEPARATE_COHORT:
                bucket = self.cohorts.setdefault(cohort, {})
                if record.prompt_id in bucket:
                    raise ValueError(
                        f"cohort {cohort!r} already holds {record.prompt_id!r}"
                    )
                bucket[record.prompt_id] = record
                self._adopt(record)
                report["cohort_records"] += 1
                continue
            existing = self.outcomes.get(record.prompt_id)
            if existing is None:
                self.outcomes[record.prompt_id] = record
                self._adopt(record)
                report["added"] += 1
                continue
            if mode == MERGE_NEW_ONLY:
                report["skipped"] += 1
                continue
            if mode == MERGE_FILL_NOT_STARTED:
                self.outcomes[record.prompt_id] = self._fill_not_started(existing, record)
                report["filled"] += 1
                continue
            # exact_continuation
            if record.planned_trials != existing.planned_trials:
                raise ValueError(
                    f"exact continuation for {record.prompt_id} changed planned_trials "
                    f"{existing.planned_trials} -> {record.planned_trials}"
                )
            _check_trajectory_evidence(existing, record)
            if record.n_correct < existing.n_correct or (
                record.counts[OUTCOME_WRONG] < existing.counts[OUTCOME_WRONG]
            ):
                raise ValueError(
                    f"exact continuation for {record.prompt_id} lost already resolved trials; "
                    f"use '{MERGE_SEPARATE_COHORT}' for an independent re-run"
                )
            merged_evidence = dict(existing.slot_evidence)
            merged_evidence.update(record.slot_evidence)
            record.slot_evidence = merged_evidence
            self.outcomes[record.prompt_id] = record
            report["continued"] += 1
        return report

    @staticmethod
    def _fill_not_started(existing: PromptOutcome, delta: PromptOutcome) -> PromptOutcome:
        """미시작 slot만 채웁니다. 기존 완료 기록(X 포함)을 대체하지 않습니다."""
        available = existing.n_not_started
        if delta.planned_trials > available:
            raise ValueError(
                f"{existing.prompt_id!r}: cannot fill {delta.planned_trials} slot(s); only "
                f"{available} slot(s) were not_started. An independent re-run of completed "
                f"slots must use '{MERGE_SEPARATE_COHORT}'"
            )
        if delta.counts[OUTCOME_NOT_STARTED] > delta.planned_trials:
            raise ValueError(f"{existing.prompt_id!r}: delta record is internally inconsistent")
        added = {
            code: int(delta.counts[code]) for code in OUTCOMES if code != OUTCOME_NOT_STARTED
        }
        n_added = sum(added.values())
        counts = _add_counts(existing.counts, added)
        # 채운 만큼 not_started를 줄입니다.
        counts[OUTCOME_NOT_STARTED] = existing.n_not_started - n_added
        overlap = set(existing.slot_evidence) & set(delta.slot_evidence)
        if overlap:
            raise ValueError(
                f"{existing.prompt_id!r}: delta reuses slot id(s) {sorted(overlap)} that are "
                "already recorded"
            )
        evidence = dict(existing.slot_evidence)
        evidence.update(delta.slot_evidence)
        return PromptOutcome(
            prompt_id=existing.prompt_id,
            counts=counts,
            planned_trials=existing.planned_trials,
            completed_trials=existing.completed_trials + n_added,
            termination_reason=delta.termination_reason,
            policy_hash=existing.policy_hash,
            scorer_version=existing.scorer_version,
            slot_evidence=evidence,
        )

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
            "scorer_version": self.scorer_version,
            "cohorts": {name: len(bucket) for name, bucket in self.cohorts.items()},
        }

    def save(self, path: str | Path) -> Path:
        """lock을 잡고 atomic하게 저장합니다."""
        from .runtime import atomic_write_json, store_lock

        path = Path(path)
        payload = {
            "policy_hash": self.policy_hash,
            "scorer_version": self.scorer_version,
            "outcomes": {k: v.as_dict() for k, v in self.outcomes.items()},
            "cohorts": {
                name: {k: v.as_dict() for k, v in bucket.items()}
                for name, bucket in self.cohorts.items()
            },
        }
        with store_lock(path):
            atomic_write_json(path, payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> OutcomeStore:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            outcomes={
                k: PromptOutcome.from_dict(v) for k, v in payload.get("outcomes", {}).items()
            },
            policy_hash=payload.get("policy_hash", ""),
            scorer_version=payload.get("scorer_version", ""),
            cohorts={
                name: {k: PromptOutcome.from_dict(v) for k, v in bucket.items()}
                for name, bucket in payload.get("cohorts", {}).items()
            },
        )

    @classmethod
    def from_jsonl(
        cls, path: str | Path, *, fill_missing_as_not_started: bool = False
    ) -> list[PromptOutcome]:
        """서버에서 만든 outcome JSONL을 읽습니다.

        기본값은 엄격한 검증입니다. `fill_missing_as_not_started=True`는 명시적인 import
        규칙으로만 씁니다.
        """
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return [
            PromptOutcome.from_dict(
                json.loads(line), fill_missing_as_not_started=fill_missing_as_not_started
            )
            for line in lines
            if line.strip()
        ]


def build_label_store(
    outcomes: OutcomeStore,
    pairs: list,
    robust_policy: BinaryRobustPolicy | None = None,
    page_variant_ids: set[str] | None = None,
) -> LabelStore:
    """outcome store와 **frozen candidate manifest 전체**에서 LabelStore를 만듭니다.

    pairs의 각 원소는 original_id / variant_id / panel_id / semantic_valid를 가진 객체여야
    합니다 (`adapters.perturbation.PairCandidate`).

    규칙:
      - `expected_members`는 manifest 전체에서 만듭니다. 측정 기록이 없는 variant도 기대
        구성원으로 남겨 partial panel이 complete로 표시되지 않게 합니다.
      - Page 존재 / outcome 존재 / score 확정 coverage를 구분해 기록합니다.
      - original_id / panel_id / variant_id의 대응과 중복을 검사합니다.
      - C4 필터를 적용하지 않고 성공·실패·성능 유지·개선 사례를 모두 보존합니다.
    """
    policy = robust_policy or BinaryRobustPolicy()
    policy.check()
    store = LabelStore(policy_hash=outcomes.policy_hash, scorer_version=outcomes.scorer_version)

    # ---- manifest 정합성 검사 ----
    seen_variants: dict[str, str] = {}
    panel_of_original: dict[str, str] = {}
    by_original: dict[str, list] = {}
    for candidate in pairs:
        variant_id = candidate.variant_id
        if variant_id in seen_variants:
            raise ValueError(
                f"duplicate variant_id {variant_id!r} in the frozen manifest "
                f"(originals {seen_variants[variant_id]!r} and {candidate.original_id!r})"
            )
        seen_variants[variant_id] = candidate.original_id
        if variant_id == candidate.original_id:
            raise ValueError(f"variant_id {variant_id!r} must differ from its original_id")
        previous_panel = panel_of_original.setdefault(candidate.original_id, candidate.panel_id)
        if previous_panel != candidate.panel_id:
            raise ValueError(
                f"original {candidate.original_id!r} maps to two panel ids "
                f"{previous_panel!r} and {candidate.panel_id!r}"
            )
        by_original.setdefault(candidate.original_id, []).append(candidate)

    for original_id, candidates in by_original.items():
        original_outcome = outcomes.outcomes.get(original_id)
        panel_id = panel_of_original[original_id]
        # expected_members는 manifest 전체입니다 (outcome 유무와 무관).
        expected = [candidate.variant_id for candidate in candidates]
        resolved_members, outcome_members, page_members, labels = [], [], [], []
        for candidate in candidates:
            variant_id = candidate.variant_id
            variant_outcome = outcomes.outcomes.get(variant_id)
            if page_variant_ids is None or variant_id in page_variant_ids:
                page_members.append(variant_id)
            if variant_outcome is not None:
                outcome_members.append(variant_id)
            if original_outcome is None or variant_outcome is None:
                missing = "original" if original_outcome is None else "variant"
                label = PairLabel(
                    original_id=original_id,
                    variant_id=variant_id,
                    panel_id=panel_id,
                    semantic_valid=candidate.semantic_valid,
                    policy_hash=outcomes.policy_hash,
                    scorer_version=outcomes.scorer_version,
                    exclusion_reason=f"missing_{missing}_outcome",
                )
                store.pairs[variant_id] = label
                labels.append(label)
                continue
            label = pair_drop(
                original_outcome,
                variant_outcome,
                original_id=original_id,
                variant_id=variant_id,
                panel_id=panel_id,
                semantic_valid=candidate.semantic_valid,
                label_source=candidate.source_dataset,
                label_version=candidate.source_revision,
            )
            store.pairs[variant_id] = label
            labels.append(label)
            if label.has_drop:
                resolved_members.append(variant_id)
        coverage = PanelCoverage(
            expected_members=expected,
            actual_members=resolved_members,
            page_members=page_members,
            outcome_members=outcome_members,
        )
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


def check_policy_consistency(labels: LabelStore) -> None:
    """서로 다른 model/thinking/sampling policy의 label을 섞지 않도록 막습니다."""
    hashes = {label.policy_hash for label in labels.pairs.values() if label.policy_hash}
    hashes |= {panel.policy_hash for panel in labels.panels.values() if panel.policy_hash}
    if len(hashes) > 1:
        raise ValueError(
            "label store mixes different model/thinking/sampling policies: "
            f"{sorted(hashes)}; keep one policy per label store"
        )
    versions = {label.scorer_version for label in labels.pairs.values() if label.scorer_version}
    if len(versions) > 1:
        raise ValueError(
            f"label store mixes scorer versions {sorted(versions)}; do not overwrite earlier "
            "results with a new scorer"
        )
