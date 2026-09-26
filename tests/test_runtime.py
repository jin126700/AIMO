"""Run lock, atomic save, config guard, dedup/resume, GPU 예산 watchdog test."""

from __future__ import annotations

import json
import os

import pytest
import torch

from aimo.runtime import (
    ConfigMismatchError,
    DedupLedger,
    GpuBudget,
    GpuBudgetExceeded,
    RunLockError,
    atomic_save,
    atomic_write_json,
    guard_config,
    mock_gpu_worker,
    run_budgeted,
    run_lock,
)


def test_run_lock_blocks_a_second_live_holder(tmp_path):
    with run_lock(tmp_path):
        assert (tmp_path / "run.lock").exists()
        holder = json.loads((tmp_path / "run.lock").read_text())
        assert holder["pid"] == os.getpid()
        # 같은 pid는 재진입을 허용하지만, 살아 있는 다른 pid는 막습니다.
        atomic_write_json(tmp_path / "run.lock", {"pid": 1, "acquired_at": 0})
        with pytest.raises(RunLockError, match="locked by live pid"):
            with run_lock(tmp_path):
                pass
    atomic_write_json(tmp_path / "run.lock", {"pid": 1, "acquired_at": 0})
    with run_lock(tmp_path, force=True):  # 죽은 것을 확인한 뒤에만 force를 씁니다.
        pass


def test_run_lock_is_released_after_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with run_lock(tmp_path):
            raise RuntimeError("boom")
    assert not (tmp_path / "run.lock").exists()


def test_atomic_save_leaves_no_temporary_files(tmp_path):
    path = atomic_save({"a": torch.ones(2)}, tmp_path / "ckpt.pt")
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp.*"))
    assert torch.equal(torch.load(path)["a"], torch.ones(2))


def test_config_guard_blocks_a_mismatched_rerun(tmp_path):
    guard_config(tmp_path, {"model": "loop4"}, "hash-a")
    guard_config(tmp_path, {"model": "loop4"}, "hash-a")  # 같은 hash는 통과합니다.
    with pytest.raises(ConfigMismatchError, match="use a new run_id"):
        guard_config(tmp_path, {"model": "loop1"}, "hash-b")


def test_dedup_ledger_survives_a_restart(tmp_path):
    ledger = DedupLedger.open(tmp_path)
    ledger.mark("pair-0001", {"outcome": "C"})
    ledger.mark("pair-0001")  # 중복 기록은 무시합니다.
    assert len(ledger) == 1
    reopened = DedupLedger.open(tmp_path)
    assert reopened.seen("pair-0001")
    assert not reopened.seen("pair-0002")
    assert len(reopened) == 1


def test_dedup_ledger_ignores_a_truncated_last_line(tmp_path):
    ledger = DedupLedger.open(tmp_path)
    ledger.mark("a")
    with (tmp_path / "ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"key": "b"')  # 잘린 줄
    reopened = DedupLedger.open(tmp_path)
    assert reopened.seen("a") and not reopened.seen("b")


def test_gpu_budget_stops_at_the_hard_limit_and_blocks_new_work(tmp_path):
    """mock worker로 175분 신규 시작 차단과 180분 종료를 CPU에서 확인합니다."""
    budget = GpuBudget.open(tmp_path, block_minutes=175, stop_minutes=180, clock=lambda: 0.0)
    step, cost = mock_gpu_worker(seconds_per_step=60 * 10)  # step 하나가 10분
    result = run_budgeted(step, budget, max_steps=100, seconds_per_step=cost)
    assert result.stopped_by_budget
    assert result.steps_done == 18  # 18 * 10 = 180분에서 멈춥니다.
    assert result.active_seconds == pytest.approx(180 * 60)
    can_start, reason = budget.can_start()
    assert not can_start and "blocked" in reason
    with pytest.raises(GpuBudgetExceeded):
        run_budgeted(step, budget, max_steps=1, seconds_per_step=cost)


def test_gpu_budget_is_kept_across_resume(tmp_path):
    budget = GpuBudget.open(tmp_path, block_minutes=175, stop_minutes=180, clock=lambda: 0.0)
    step, cost = mock_gpu_worker(seconds_per_step=60 * 60)
    run_budgeted(step, budget, max_steps=2, seconds_per_step=cost)  # 120분 소비
    reopened = GpuBudget.open(tmp_path, block_minutes=175, stop_minutes=180, clock=lambda: 0.0)
    assert reopened.active_seconds() == pytest.approx(120 * 60)
    can_start, _ = reopened.can_start()
    assert can_start  # 175분 미만이므로 재개할 수 있습니다.
    result = run_budgeted(step, reopened, max_steps=5, seconds_per_step=cost)
    assert result.stopped_by_budget
    assert reopened.active_seconds() == pytest.approx(180 * 60)


def test_budget_blocks_a_new_start_at_the_block_threshold(tmp_path):
    budget = GpuBudget.open(tmp_path, block_minutes=175, stop_minutes=180, clock=lambda: 0.0)
    budget.add(175 * 60)
    can_start, reason = budget.can_start()
    assert not can_start
    assert "175" in reason
