"""행동 측정(collection) runner.

실제 generation backend를 의존성으로 받아, slot 단위 실행·dedup/resume·예산·오류 전달을
담당합니다. backend를 주입할 수 있으므로 로컬에서는 실제 weights 없이 mock으로 전체 경로를
검증합니다 (`adapters.qwen.MockGenerationBackend`).

여기서 하는 일:
  - 필요한 config(thinking profile calibration) 검증
  - slot ledger로 dedup / resume (이미 끝난 slot은 다시 실행하지 않음)
  - 총 context(prompt + generated) 검사
  - slot outcome 분류와 PromptOutcome 집계 (slot 단위 trajectory 증거 포함)
  - 예산 초과·중단 요청 전달과 부분 결과 보존
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .adapters.qwen import (
    SCORER_VERSION,
    SlotRequest,
    check_total_context,
    classify_thinking_slot,
)
from .labels import OUTCOME_NOT_STARTED, OUTCOMES, PromptOutcome
from .runtime import BudgetGuard, DedupLedger


@dataclass
class CollectionPlan:
    """prompt별 실행 계획. planned slot 수와 slot id를 미리 고정합니다."""

    prompt_id: str
    prompt: str
    gold: str
    slot_ids: list[str]
    seeds: list[int] = field(default_factory=list)
    is_final_cap: bool = False
    thinking_already_open: bool = False

    def __post_init__(self) -> None:
        if not self.slot_ids:
            raise ValueError(f"{self.prompt_id!r}: planned slot list must not be empty")
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError(f"{self.prompt_id!r}: duplicate slot ids in the plan")
        if not self.seeds:
            self.seeds = list(range(len(self.slot_ids)))
        if len(self.seeds) != len(self.slot_ids):
            raise ValueError(f"{self.prompt_id!r}: seeds and slot_ids must have the same length")

    @property
    def planned_trials(self) -> int:
        return len(self.slot_ids)


@dataclass
class CollectionReport:
    """실행 요약. 부분 결과도 그대로 보고합니다."""

    outcomes: list[PromptOutcome]
    executed_slots: int = 0
    skipped_slots: int = 0
    stopped: bool = False
    stop_reason: str = ""
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_prompts": len(self.outcomes),
            "executed_slots": self.executed_slots,
            "skipped_slots": self.skipped_slots,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "errors": self.errors,
        }


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def token_prefix_hash(token_ids: list[int] | None) -> str:
    """생성된 token prefix의 hash. trajectory 동일성 증거로 씁니다."""
    if not token_ids:
        return ""
    blob = json.dumps(token_ids[:32]).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def collect_outcomes(
    backend,
    plans: list[CollectionPlan],
    profile,
    *,
    ledger: DedupLedger | None = None,
    guard: BudgetGuard | None = None,
    policy_hash: str = "",
    scorer_version: str = SCORER_VERSION,
    special_token_ids: frozenset[int] | set[int] = frozenset(),
) -> CollectionReport:
    """계획된 slot을 실행해 PromptOutcome 목록을 만듭니다.

    이미 ledger에 기록된 slot은 다시 실행하지 않고 저장된 outcome을 재사용합니다. 예산
    초과나 중단 요청이 오면 남은 slot을 `not_started`로 남기고 부분 결과를 돌려줍니다.
    """
    pending = profile.needs_calibration()
    if pending:
        raise ValueError(
            f"thinking profile is not calibrated yet ({pending}); collection must not start "
            "with an unfrozen protocol"
        )
    report = CollectionReport(outcomes=[])
    stopped = False
    for plan in plans:
        counts = dict.fromkeys(OUTCOMES, 0)
        completed = 0
        evidence: dict[str, dict] = {}
        for slot_id, seed in zip(plan.slot_ids, plan.seeds, strict=True):
            if stopped:
                counts[OUTCOME_NOT_STARTED] += 1
                continue
            if guard is not None:
                hit, why = guard.should_stop()
                if hit:
                    stopped = True
                    report.stopped = True
                    report.stop_reason = why
                    counts[OUTCOME_NOT_STARTED] += 1
                    continue
            cached = _cached_slot(ledger, slot_id)
            if cached is not None:
                counts[cached["outcome"]] += 1
                completed += 1 if cached["outcome"] != OUTCOME_NOT_STARTED else 0
                evidence[slot_id] = cached["evidence"]
                report.skipped_slots += 1
                continue
            request = SlotRequest(
                prompt_id=plan.prompt_id,
                slot_id=slot_id,
                prompt=plan.prompt,
                gold=plan.gold,
                seed=seed,
                is_final_cap=plan.is_final_cap,
                thinking_already_open=plan.thinking_already_open,
            )
            try:
                result = backend.generate(request)
            except Exception as exc:  # noqa: BLE001 - backend 오류를 그대로 전달합니다
                report.errors.append(f"{slot_id}: {type(exc).__name__}: {exc}")
                counts["infra_error"] += 1
                completed += 1
                continue
            if result.started and not result.infra_error:
                # 총 context는 prompt + generated로 검사합니다 (조용히 truncate하지 않습니다).
                try:
                    check_total_context(
                        result.prompt_tokens,
                        result.generated_tokens,
                        profile.max_total_context,
                    )
                except ValueError as exc:
                    report.errors.append(f"{slot_id}: {exc}")
                    counts["infra_error"] += 1
                    completed += 1
                    continue
            outcome = classify_thinking_slot(
                started=result.started,
                infra_error=result.infra_error,
                hit_cap=result.hit_cap,
                is_final_cap=plan.is_final_cap,
                text=result.text,
                gold=plan.gold,
                thinking_already_open=result.thinking_already_open
                or plan.thinking_already_open,
                generated_token_ids=result.generated_token_ids,
                special_token_ids=special_token_ids,
            )
            counts[outcome] += 1
            if outcome != OUTCOME_NOT_STARTED:
                completed += 1
            slot_evidence = {
                "request_id": slot_id,
                "seed": seed,
                "prompt_hash": prompt_hash(plan.prompt),
                "policy_hash": policy_hash,
                "token_prefix_hash": token_prefix_hash(result.generated_token_ids),
            }
            evidence[slot_id] = slot_evidence
            report.executed_slots += 1
            if ledger is not None:
                ledger.mark(slot_id, {"outcome": outcome, "evidence": slot_evidence})
        report.outcomes.append(
            PromptOutcome(
                prompt_id=plan.prompt_id,
                counts=counts,
                planned_trials=plan.planned_trials,
                completed_trials=completed,
                termination_reason="stopped" if stopped else "completed",
                policy_hash=policy_hash,
                scorer_version=scorer_version,
                slot_evidence=evidence,
            )
        )
    return report


def _cached_slot(ledger: DedupLedger | None, slot_id: str) -> dict | None:
    """ledger에 이미 기록된 slot 결과를 돌려줍니다 (resume용)."""
    if ledger is None or not ledger.seen(slot_id):
        return None
    payload = ledger.payload(slot_id)
    if not payload or "outcome" not in payload:
        return None
    return {"outcome": payload["outcome"], "evidence": payload.get("evidence", {})}
