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
    backend, plans, profile, *, ledger=None, guard=None, policy_hash="",
    scorer_version=SCORER_VERSION, special_token_ids=frozenset(),
):
    """Durable slot state: started -> raw evidence -> completed ledger; never retry started slots."""
    from dataclasses import asdict
    from pathlib import Path
    from .runtime import atomic_write_json, store_lock
    from .adapters.qwen import SlotResult
    if profile.needs_calibration():
        raise ValueError("thinking profile not calibrated")
    report = CollectionReport(outcomes=[])
    all_ids = [sid for p in plans for sid in p.slot_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Slot IDs must be globally unique")
    def stop():
        return guard is not None and guard.should_stop()[0]
    backend.stop_check = stop
    for plan in plans:
        counts = dict.fromkeys(OUTCOMES, 0)
        evidence = {}
        for sid, seed in zip(plan.slot_ids, plan.seeds, strict=True):
            identity = {"request_id": sid, "seed": seed, "prompt_hash": prompt_hash(plan.prompt),
                        "policy_hash": policy_hash, "scorer_version": scorer_version,
                        "model_hash": getattr(backend, "model_hash", "mock"),
                        "gold_hash": prompt_hash(plan.gold)}
            rendered = backend.render(plan.prompt) if hasattr(backend, "render") else None
            if rendered is not None:
                identity["input_token_hash"] = hashlib.sha256(
                    json.dumps(rendered["input_ids"][0].tolist()).encode()).hexdigest()
            root = ledger.path.parent / "slots" if ledger is not None else None
            path = root / (hashlib.sha256(sid.encode()).hexdigest() + ".json") if root else None
            def work():
                if path is not None and path.exists():
                    raw = json.loads(path.read_text())
                    if raw["identity"] != identity:
                        raise ValueError("Cached slot identity mismatch: " + sid)
                    if raw["state"] == "started":
                        raw.update(state="interrupted", outcome="U_score", termination="interrupted")
                        atomic_write_json(path, raw)
                    report.skipped_slots += 1
                    return raw
                if ledger is not None and ledger.seen(sid):
                    raise ValueError("Legacy ledger without durable evidence: " + sid)
                if stop():
                    report.stopped = True
                    report.stop_reason = guard.should_stop()[1]
                    return {"outcome": "not_started", "identity": identity}
                raw = {"identity": identity, "state": "started"}
                if path is not None:
                    atomic_write_json(path, raw)
                request = SlotRequest(plan.prompt_id, sid, plan.prompt, plan.gold, seed,
                                      plan.is_final_cap, plan.thinking_already_open)
                try:
                    result = backend.generate(request)
                    check_total_context(result.prompt_tokens, result.generated_tokens,
                                        profile.max_total_context)
                except Exception as exc:
                    result = SlotResult(sid, infra_error=True, error_message=str(exc),
                                        termination_reason="infra_error")
                    report.errors.append(sid + ": " + str(exc))
                outcome = classify_thinking_slot(
                    started=result.started, infra_error=result.infra_error, hit_cap=result.hit_cap,
                    is_final_cap=plan.is_final_cap, text=result.text, gold=plan.gold,
                    thinking_already_open=result.thinking_already_open or plan.thinking_already_open,
                    generated_token_ids=result.generated_token_ids, special_token_ids=special_token_ids)
                if result.termination_reason == "external_stop":
                    outcome = "U_score"
                raw.update(state="completed", outcome=outcome, result=asdict(result),
                           termination=result.termination_reason,
                           scorer={"gold": plan.gold, "version": scorer_version,
                                   "outcome": outcome})
                if path is not None:
                    atomic_write_json(path, raw)
                if ledger is not None:
                    ledger.mark(sid, {"outcome": outcome, "evidence": identity,
                                      "artifact": str(path)})
                report.executed_slots += 1
                return raw
            if path is not None:
                with store_lock(path):
                    raw = work()
            else:
                raw = work()
            counts[raw["outcome"]] += 1
            evidence[sid] = {**{k: identity[k] for k in ("request_id", "seed", "prompt_hash", "policy_hash")}, "token_prefix_hash": token_prefix_hash(
                raw.get("result", {}).get("generated_token_ids"))}
        report.outcomes.append(PromptOutcome(
            prompt_id=plan.prompt_id, counts=counts, planned_trials=plan.planned_trials,
            completed_trials=plan.planned_trials-counts["not_started"],
            termination_reason="stopped" if report.stopped else "completed",
            policy_hash=policy_hash, scorer_version=scorer_version, slot_evidence=evidence))
    return report
