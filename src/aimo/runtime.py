"""Run 관리: lock, atomic save, config mismatch 차단, dedup/resume, GPU 예산 watchdog.

GPU 예산 watchdog은 이 실행이 직접 시작한 worker만 협조적으로 멈춥니다. 다른
사용자의 process나 container를 종료하는 코드는 포함하지 않습니다.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

LOCK_NAME = "run.lock"
CONFIG_NAME = "config.json"
LEDGER_NAME = "ledger.jsonl"
BUDGET_NAME = "gpu_budget.json"


# --------------------------------------------------------------------------------------
# Atomic 저장
# --------------------------------------------------------------------------------------


def atomic_write_text(path: str | Path, text: str) -> Path:
    """같은 directory의 tmp file에 쓴 뒤 os.replace로 교체합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def atomic_save(obj: Any, path: str | Path) -> Path:
    """torch.save의 atomic 버전. 중간에 죽어도 기존 checkpoint가 남습니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------------------
# Run lock
# --------------------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLockError(RuntimeError):
    pass


@contextmanager
def run_lock(run_dir: str | Path, *, force: bool = False) -> Iterator[Path]:
    """한 run directory에 대해 하나의 실행만 허용합니다."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / LOCK_NAME
    if lock_path.exists():
        try:
            holder = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            holder = {"pid": -1}
        pid = int(holder.get("pid", -1))
        if pid > 0 and _pid_alive(pid) and pid != os.getpid() and not force:
            raise RunLockError(
                f"run directory {run_dir} is locked by live pid {pid}; "
                "wait for it to finish or pass force=True after confirming it is dead"
            )
        lock_path.unlink(missing_ok=True)
    atomic_write_json(lock_path, {"pid": os.getpid(), "acquired_at": time.time()})
    try:
        yield lock_path
    finally:
        lock_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------
# Config mismatch 차단
# --------------------------------------------------------------------------------------


class ConfigMismatchError(RuntimeError):
    pass


def guard_config(run_dir: str | Path, config_payload: dict, config_hash: str) -> dict:
    """run directory에 config를 고정합니다. 이후 hash가 다르면 실행을 막습니다."""
    run_dir = Path(run_dir)
    path = run_dir / CONFIG_NAME
    record = {"config_hash": config_hash, "config": config_payload}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            raise ConfigMismatchError(
                f"run directory {run_dir} was created with config hash "
                f"{previous.get('config_hash')} but this run has {config_hash}; "
                "use a new run_id instead of mixing configs"
            )
        return previous
    atomic_write_json(path, record)
    return record


# --------------------------------------------------------------------------------------
# Dedup / resume ledger
# --------------------------------------------------------------------------------------


@dataclass
class DedupLedger:
    """완료한 작업 key를 append-only jsonl로 남겨 resume 시 중복을 막습니다."""

    path: Path
    _seen: set[str] = field(default_factory=set)

    @classmethod
    def open(cls, run_dir: str | Path, name: str = LEDGER_NAME) -> DedupLedger:
        path = Path(run_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    seen.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue  # 잘린 마지막 줄은 무시하고 다시 처리합니다.
        return cls(path=path, _seen=seen)

    def seen(self, key: str) -> bool:
        return key in self._seen

    def mark(self, key: str, payload: dict | None = None) -> None:
        if key in self._seen:
            return
        record = {"key": key, "at": time.time()}
        if payload:
            record["payload"] = payload
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._seen.add(key)

    def __len__(self) -> int:
        return len(self._seen)


# --------------------------------------------------------------------------------------
# GPU 예산 watchdog
# --------------------------------------------------------------------------------------


class GpuBudgetExceeded(RuntimeError):
    pass


@dataclass
class GpuBudget:
    """GPU-active 누적 시간을 파일로 관리합니다.

    누적 block_minutes 이상이면 새 작업 시작을 막고, 진행 중이라도 stop_minutes에
    도달하면 협조적으로 종료합니다. resume해도 같은 파일을 읽어 예산이 유지됩니다.
    clock은 test에서 주입할 수 있습니다.
    """

    path: Path
    block_minutes: float = 175.0
    stop_minutes: float = 180.0
    clock: Callable[[], float] = time.monotonic

    @classmethod
    def open(
        cls,
        run_dir: str | Path,
        block_minutes: float = 175.0,
        stop_minutes: float = 180.0,
        clock: Callable[[], float] = time.monotonic,
        name: str = BUDGET_NAME,
    ) -> GpuBudget:
        return cls(
            path=Path(run_dir) / name,
            block_minutes=block_minutes,
            stop_minutes=stop_minutes,
            clock=clock,
        )

    def active_seconds(self) -> float:
        if not self.path.exists():
            return 0.0
        return float(json.loads(self.path.read_text(encoding="utf-8")).get("active_seconds", 0.0))

    def _write(self, seconds: float) -> None:
        atomic_write_json(
            self.path, {"active_seconds": seconds, "updated_at": time.time()}
        )

    def add(self, seconds: float) -> float:
        total = self.active_seconds() + max(seconds, 0.0)
        self._write(total)
        return total

    def can_start(self) -> tuple[bool, str]:
        used = self.active_seconds() / 60.0
        if used >= self.block_minutes:
            return False, (
                f"GPU-active budget exhausted: {used:.1f} min used, "
                f"new work blocked at {self.block_minutes:.0f} min"
            )
        return True, f"{used:.1f} min of GPU-active time used"

    def remaining_stop_seconds(self) -> float:
        return self.stop_minutes * 60.0 - self.active_seconds()


@dataclass
class BudgetedRun:
    steps_done: int
    stopped_by_budget: bool
    active_seconds: float
    reason: str


def run_budgeted(
    step_fn: Callable[[int], None],
    budget: GpuBudget,
    max_steps: int,
    seconds_per_step: Callable[[], float] | None = None,
) -> BudgetedRun:
    """step_fn을 예산 안에서 반복 호출합니다.

    step 사이에서만 중단을 판단하므로 외부 process를 강제 종료하지 않습니다.
    seconds_per_step이 주어지면 그 값으로 GPU-active 시간을 적립합니다 (mock worker
    CPU 검증용). 없으면 실제 벽시계 시간을 씁니다.
    """
    ok, reason = budget.can_start()
    if not ok:
        raise GpuBudgetExceeded(reason)
    done = 0
    stopped = False
    for i in range(max_steps):
        if budget.remaining_stop_seconds() <= 0:
            stopped = True
            reason = (
                f"stopped at {budget.active_seconds() / 60.0:.1f} min GPU-active time "
                f"(hard stop {budget.stop_minutes:.0f} min)"
            )
            break
        started = budget.clock()
        step_fn(i)
        elapsed = seconds_per_step() if seconds_per_step else budget.clock() - started
        budget.add(elapsed)
        done += 1
    if not stopped:
        reason = f"completed {done} step(s) within budget"
    return BudgetedRun(
        steps_done=done,
        stopped_by_budget=stopped,
        active_seconds=budget.active_seconds(),
        reason=reason,
    )


def mock_gpu_worker(seconds_per_step: float) -> tuple[Callable[[int], None], Callable[[], float]]:
    """CPU에서 watchdog을 검증하기 위한 mock worker.

    실제 GPU를 쓰지 않고 step마다 seconds_per_step만큼 GPU-active 시간을 적립합니다.
    """
    log: list[int] = []

    def step(i: int) -> None:
        log.append(i)

    def cost() -> float:
        return seconds_per_step

    step.log = log  # type: ignore[attr-defined]
    return step, cost
