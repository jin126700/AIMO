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
def store_lock(path: str | Path, *, timeout: float = 10.0) -> Iterator[Path]:
    """store 파일 하나에 대한 배타 lock.

    O_CREAT|O_EXCL로 lock 파일을 만들고, 살아 있는 다른 pid가 쥐고 있으면 timeout까지
    기다립니다. 다른 사용자의 process를 종료하지 않습니다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(handle, str(os.getpid()).encode())
            os.close(handle)
            break
        except FileExistsError:
            holder = -1
            try:
                holder = int(lock_path.read_text(encoding="utf-8").strip() or -1)
            except (OSError, ValueError):
                holder = -1
            if holder > 0 and not _pid_alive(holder):
                lock_path.unlink(missing_ok=True)  # 죽은 holder의 lock만 회수합니다.
                continue
            if time.monotonic() >= deadline:
                raise RunLockError(
                    f"store {path} is locked by pid {holder}; retry after it finishes"
                ) from None
            time.sleep(0.05)
    try:
        yield path
    finally:
        lock_path.unlink(missing_ok=True)


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
    _payloads: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def open(cls, run_dir: str | Path, name: str = LEDGER_NAME) -> DedupLedger:
        path = Path(run_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        payloads: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    seen.add(record["key"])
                    if record.get("payload") is not None:
                        payloads[record["key"]] = record["payload"]
                except (json.JSONDecodeError, KeyError):
                    continue  # 잘린 마지막 줄은 무시하고 다시 처리합니다.
        return cls(path=path, _seen=seen, _payloads=payloads)

    def seen(self, key: str) -> bool:
        return key in self._seen

    def payload(self, key: str) -> dict | None:
        """저장된 payload. resume에서 이전 결과를 재사용할 때 씁니다."""
        return self._payloads.get(key)

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
        if payload:
            self._payloads[key] = payload

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


class StopRequest:
    """협조적 중단 신호.

    긴 작업 도중에도 `requested`를 polling해 스스로 멈출 수 있게 합니다. 다른 사용자의
    process나 container를 종료하지 않습니다.
    """

    def __init__(self) -> None:
        self._requested = False
        self._reason = ""

    def request(self, reason: str = "stop requested") -> None:
        self._requested = True
        self._reason = reason

    @property
    def requested(self) -> bool:
        return self._requested

    @property
    def reason(self) -> str:
        return self._reason

    def __call__(self) -> bool:
        return self._requested


class BudgetGuard:
    """긴 작업에 예산과 중단 신호를 연결합니다.

    - `start()`에서 신규 시작을 막습니다 (누적 block_minutes 이상).
    - context를 벗어날 때 **예외가 나도** elapsed를 저장합니다 (finally).
    - `should_stop()`은 hard stop 도달 또는 외부 중단 요청을 알려 줍니다. step 사이 검사만으로
      hard stop을 보장한다고 쓰지 않습니다: 작업 쪽에서 충분히 자주 polling해야 합니다.
    - 소유한 subprocess는 `terminate_owned`로만 종료합니다.
    """

    def __init__(
        self,
        budget: GpuBudget,
        stop: StopRequest | None = None,
        seconds_per_step: Callable[[], float] | None = None,
    ) -> None:
        self.budget = budget
        self.stop = stop or StopRequest()
        self.seconds_per_step = seconds_per_step
        self._started_at: float | None = None
        self._owned: list = []

    def start(self) -> str:
        ok, reason = self.budget.can_start()
        if not ok:
            raise GpuBudgetExceeded(reason)
        self._started_at = self.budget.clock()
        return reason

    def elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        if self.seconds_per_step is not None:
            return self.seconds_per_step()
        return self.budget.clock() - self._started_at

    def commit(self) -> float:
        """지금까지의 elapsed를 예산에 적립하고 구간을 다시 엽니다."""
        elapsed = self.elapsed()
        if elapsed > 0:
            self.budget.add(elapsed)
        self._started_at = self.budget.clock()
        return elapsed

    def should_stop(self) -> tuple[bool, str]:
        if self.stop.requested:
            return True, self.stop.reason
        # mock clock에서는 seconds_per_step이 '다음 step의 비용'이므로 경과로 빼지 않습니다.
        pending = self.elapsed() if self.seconds_per_step is None else 0.0
        if self.budget.remaining_stop_seconds() - pending <= 0:
            return True, (
                f"hard stop at {self.budget.stop_minutes:.0f} min GPU-active time "
                f"(used {self.budget.active_seconds() / 60.0:.1f} min)"
            )
        return False, ""

    def own(self, process) -> None:
        """이 실행이 직접 시작한 subprocess만 등록합니다."""
        self._owned.append(process)

    def terminate_owned(self, grace: float = 5.0) -> list[int]:
        """등록된(소유한) subprocess만 종료합니다. 다른 process는 건드리지 않습니다."""
        stopped = []
        for process in self._owned:
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=grace)
            except Exception:  # noqa: BLE001 - subprocess 구현에 따라 예외가 다릅니다
                process.kill()
            stopped.append(process.pid)
        self._owned.clear()
        return stopped

    def __enter__(self) -> BudgetGuard:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # 예외·중단 시에도 elapsed를 반드시 저장합니다.
        try:
            self.commit()
        finally:
            self.terminate_owned()
            self._started_at = None


def run_budgeted(
    step_fn: Callable[[int], None],
    budget: GpuBudget,
    max_steps: int,
    seconds_per_step: Callable[[], float] | None = None,
    stop: StopRequest | None = None,
) -> BudgetedRun:
    """step_fn을 예산 안에서 반복 호출합니다.

    step 사이에서 중단을 판단하며 외부 process를 강제 종료하지 않습니다. 예외가 나도
    지금까지의 elapsed는 예산에 적립됩니다.
    seconds_per_step이 주어지면 그 값으로 GPU-active 시간을 적립합니다 (mock worker
    CPU 검증용). 없으면 실제 벽시계 시간을 씁니다.
    """
    guard = BudgetGuard(budget, stop=stop, seconds_per_step=seconds_per_step)
    reason = guard.start()
    done = 0
    stopped = False
    try:
        for i in range(max_steps):
            hit, why = guard.should_stop()
            if hit:
                stopped = True
                reason = why
                break
            started = budget.clock()
            step_fn(i)
            elapsed = seconds_per_step() if seconds_per_step else budget.clock() - started
            budget.add(elapsed)
            guard._started_at = budget.clock()  # 적립한 구간은 다시 세지 않습니다
            done += 1
    finally:
        guard.terminate_owned()
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


# --------------------------------------------------------------------------------------
# Device 해석
# --------------------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    """config의 device 문자열을 torch.device로 바꿉니다.

    요청한 accelerator가 없으면 **명시적으로 오류**를 냅니다. CPU로 조용히 fallback하지
    않습니다.
    """
    text = (name or "cpu").strip().lower()
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {name!r} was requested but CUDA is not available in this process; "
                "refusing to fall back to CPU silently"
            )
        return torch.device(text)
    if text == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError(
                f"device {name!r} was requested but MPS is not available in this process; "
                "refusing to fall back to CPU silently"
            )
        return torch.device("mps")
    if text == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device {name!r}; use 'cpu', 'cuda[:N]' or 'mps'")
