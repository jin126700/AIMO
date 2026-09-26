"""Collection/extraction runner와 budget·prepared registry regression test."""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from aimo.config import ThinkingProfileConfig
from aimo.runtime import (
    BudgetGuard,
    DedupLedger,
    GpuBudget,
    GpuBudgetExceeded,
    RunLockError,
    StopRequest,
    store_lock,
)

transformers = pytest.importorskip("transformers", reason="optional server dependency")


def calibrated() -> ThinkingProfileConfig:
    return ThinkingProfileConfig(
        max_new_tokens=256,
        samples_per_prompt=4,
        max_total_context=1024,
        numerical_backend="bf16",
        scorer_id="aimo.thinking.exact_final_answer",
        scorer_version="2",
    )


def plan(prompt_id: str = "p1", slots: int = 4):
    from aimo.collect import CollectionPlan

    return CollectionPlan(
        prompt_id=prompt_id,
        prompt="Q?",
        gold="7",
        slot_ids=[f"{prompt_id}#s{i}" for i in range(slots)],
    )


def responses():
    from aimo.adapters.qwen import SlotResult

    return {
        "p1#s0": SlotResult(
            slot_id="p1#s0",
            text=r"<think>x</think> \boxed{7}",
            prompt_tokens=10,
            generated_tokens=20,
            generated_token_ids=[1, 2, 3],
        ),
        "p1#s1": SlotResult(
            slot_id="p1#s1",
            text=r"<think>x</think> \boxed{6}",
            prompt_tokens=10,
            generated_tokens=20,
            generated_token_ids=[1, 2],
        ),
        "p1#s2": SlotResult(
            slot_id="p1#s2",
            text="<think>unfinished",
            hit_cap=True,
            prompt_tokens=10,
            generated_tokens=256,
            generated_token_ids=[9],
        ),
        "p1#s3": SlotResult(slot_id="p1#s3", infra_error=True, error_message="oom"),
    }


# --------------------------------------------------------------------------------------
# collection runner (mock backend)
# --------------------------------------------------------------------------------------


def test_collection_requires_a_calibrated_profile():
    from aimo.adapters.qwen import MockGenerationBackend
    from aimo.collect import collect_outcomes

    with pytest.raises(ValueError, match="not calibrated"):
        collect_outcomes(MockGenerationBackend({}), [plan()], ThinkingProfileConfig())


def test_collection_classifies_and_aggregates_slots(tmp_path):
    from aimo.adapters.qwen import MockGenerationBackend
    from aimo.collect import collect_outcomes

    ledger = DedupLedger.open(tmp_path, name="slots.jsonl")
    report = collect_outcomes(
        MockGenerationBackend(responses()),
        [plan()],
        calibrated(),
        ledger=ledger,
        policy_hash="pol1",
    )
    outcome = report.outcomes[0]
    assert outcome.counts["C"] == 1
    assert outcome.counts["W"] == 1
    assert outcome.counts["X"] == 1  # 미완료는 wrong이 아닙니다
    assert outcome.counts["infra_error"] == 1
    assert outcome.planned_trials == 4
    assert report.executed_slots == 4
    # slot 단위 trajectory 증거가 남습니다.
    evidence = outcome.slot_evidence["p1#s0"]
    assert set(evidence) == {
        "request_id",
        "seed",
        "prompt_hash",
        "policy_hash",
        "token_prefix_hash",
    }


def test_collection_resumes_without_regenerating(tmp_path):
    from aimo.adapters.qwen import MockGenerationBackend
    from aimo.collect import collect_outcomes

    first_backend = MockGenerationBackend(responses())
    collect_outcomes(
        first_backend,
        [plan()],
        calibrated(),
        ledger=DedupLedger.open(tmp_path, name="slots.jsonl"),
        policy_hash="pol1",
    )
    second_backend = MockGenerationBackend(responses())
    report = collect_outcomes(
        second_backend,
        [plan()],
        calibrated(),
        ledger=DedupLedger.open(tmp_path, name="slots.jsonl"),
        policy_hash="pol1",
    )
    assert second_backend.calls == []  # 다시 생성하지 않습니다
    assert report.skipped_slots == 4
    assert report.outcomes[0].counts["C"] == 1


def test_collection_propagates_stop_and_keeps_partial_results(tmp_path):
    from aimo.adapters.qwen import MockGenerationBackend
    from aimo.collect import collect_outcomes

    stop = StopRequest()
    stop.request("외부 중단")
    guard = BudgetGuard(GpuBudget.open(tmp_path, clock=lambda: 0.0), stop=stop)
    guard.start()
    report = collect_outcomes(
        MockGenerationBackend(responses()), [plan()], calibrated(), guard=guard,
        policy_hash="pol1",
    )
    assert report.stopped and "중단" in report.stop_reason
    assert report.outcomes[0].counts["not_started"] == 4


def test_collection_rejects_context_overflow():
    from aimo.adapters.qwen import MockGenerationBackend, SlotResult
    from aimo.collect import collect_outcomes

    over = {
        "p1#s0": SlotResult(
            slot_id="p1#s0", text="ok", prompt_tokens=1000, generated_tokens=500,
            generated_token_ids=[1],
        )
    }
    report = collect_outcomes(
        MockGenerationBackend(over), [plan(slots=1)], calibrated(), policy_hash="pol1"
    )
    assert report.outcomes[0].counts["infra_error"] == 1
    assert "exceeds the configured limit" in report.errors[0]


def test_real_backend_refuses_locally():
    from aimo.adapters import AdapterUnavailable
    from aimo.adapters.qwen import QwenGenerationBackend

    with pytest.raises(AdapterUnavailable, match="SERVER_PENDING"):
        QwenGenerationBackend(ThinkingProfileConfig())  # calibration 미완
    with pytest.raises(AdapterUnavailable, match="SERVER_PENDING"):
        QwenGenerationBackend(calibrated())  # weights 없음


# --------------------------------------------------------------------------------------
# extraction runner
# --------------------------------------------------------------------------------------


def extraction_requests(n: int = 3, prompt_len: int = 30):
    from aimo.adapters.qwen import ExtractionRequest, select_landmarks

    torch.manual_seed(0)
    requests = []
    for index in range(n):
        offsets, valid, rel = select_landmarks(list(range(1, prompt_len, 2)), prompt_len)
        requests.append(
            ExtractionRequest(
                original_id="o",
                variant_id=f"o#v{index}",
                input_ids=torch.randint(0, 64, (1, prompt_len)),
                landmark_offsets=offsets,
                valid=valid,
                relative_positions=rel,
            )
        )
    return requests


def test_extraction_runner_matches_reference_values_and_is_cpu_float32():
    from aimo.adapters.qwen import build_tiny_qwen, run_page_extraction

    tiny = build_tiny_qwen(hidden_size=32, n_layers=4)
    requests = extraction_requests(1)
    pages, report = run_page_extraction(tiny.model, requests)
    page = pages[0]
    assert report["extracted"] == 1 and not report["errors"]
    assert page.state.dtype == torch.float32 and page.state.device.type == "cpu"
    assert page.residual_identity_error() < 1e-4

    # 같은 forward를 직접 hook해 landmark 값을 비교합니다.
    inner = tiny.model.model
    captured: dict[int, torch.Tensor] = {}
    attn: dict[int, torch.Tensor] = {}
    mlp: dict[int, torch.Tensor] = {}

    def pre(index):
        def hook(_module, args, kwargs):
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = args[0]
            captured[index] = hidden.detach()[0].clone()

        return hook

    def out(store, index):
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            store[index] = tensor.detach()[0].clone()

        return hook

    handles = []
    for index, layer in enumerate(inner.layers):
        handles.append(layer.register_forward_pre_hook(pre(index), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(out(attn, index)))
        handles.append(layer.mlp.register_forward_hook(out(mlp, index)))
    with torch.no_grad():
        inner(input_ids=requests[0].input_ids, use_cache=False)
    for handle in handles:
        handle.remove()
    offsets = requests[0].landmark_offsets
    for depth in range(4):
        assert torch.allclose(page.state[depth], captured[depth][offsets].float(), atol=1e-6)
        assert torch.allclose(page.updates[depth, :, 0], attn[depth][offsets].float(), atol=1e-6)
        assert torch.allclose(page.updates[depth, :, 1], mlp[depth][offsets].float(), atol=1e-6)


def test_extraction_runner_dedups_and_stops(tmp_path):
    from aimo.adapters.qwen import build_tiny_qwen, run_page_extraction

    tiny = build_tiny_qwen(hidden_size=16, n_layers=2)
    requests = extraction_requests(3, prompt_len=24)
    ledger = DedupLedger.open(tmp_path, name="ex.jsonl")
    _, first = run_page_extraction(tiny.model, requests, ledger=ledger)
    assert first["extracted"] == 3
    _, second = run_page_extraction(
        tiny.model, requests, ledger=DedupLedger.open(tmp_path, name="ex.jsonl")
    )
    assert second["extracted"] == 0 and second["skipped"] == 3

    stop = StopRequest()
    stop.request("중단")
    guard = BudgetGuard(GpuBudget.open(tmp_path, clock=lambda: 0.0), stop=stop)
    guard.start()
    _, third = run_page_extraction(tiny.model, requests, guard=guard)
    assert third["stopped"] and third["extracted"] == 0


def test_real_extraction_needs_server_weights():
    from aimo.adapters import AdapterUnavailable
    from aimo.adapters.qwen import load_real_qwen

    class Cfg:
        pass

    cfg = Cfg()
    cfg.server = Cfg()
    cfg.server.thinking = calibrated()
    cfg.server.model_id = "Qwen/Qwen3-4B"
    with pytest.raises(AdapterUnavailable, match="SERVER_PENDING"):
        load_real_qwen(cfg)


# --------------------------------------------------------------------------------------
# budget과 store lock
# --------------------------------------------------------------------------------------


def test_budget_saves_elapsed_even_on_exception(tmp_path):
    clock = {"now": 0.0}
    budget = GpuBudget.open(tmp_path, clock=lambda: clock["now"])
    with pytest.raises(RuntimeError, match="boom"):
        with BudgetGuard(budget):
            clock["now"] = 120.0
            raise RuntimeError("boom")
    assert budget.active_seconds() == pytest.approx(120.0)


def test_budget_blocks_new_work_and_stops_at_the_hard_limit(tmp_path):
    from aimo.runtime import mock_gpu_worker, run_budgeted

    budget = GpuBudget.open(tmp_path, clock=lambda: 0.0)
    step, cost = mock_gpu_worker(seconds_per_step=60 * 10)
    result = run_budgeted(step, budget, max_steps=100, seconds_per_step=cost)
    assert result.steps_done == 18  # 180분에서 종료
    assert result.stopped_by_budget
    can_start, reason = budget.can_start()
    assert not can_start and "175" in reason
    # resume해도 누적 예산이 리셋되지 않습니다.
    reopened = GpuBudget.open(tmp_path, clock=lambda: 0.0)
    assert reopened.active_seconds() == pytest.approx(180 * 60)
    with pytest.raises(GpuBudgetExceeded):
        run_budgeted(step, reopened, max_steps=1, seconds_per_step=cost)


def test_budget_terminates_only_owned_processes(tmp_path):
    import subprocess

    budget = GpuBudget.open(tmp_path, clock=lambda: 0.0)
    guard = BudgetGuard(budget)
    guard.start()
    process = subprocess.Popen(["sleep", "30"])
    guard.own(process)
    stopped = guard.terminate_owned(grace=5.0)
    assert stopped == [process.pid]
    assert process.poll() is not None
    assert guard.terminate_owned() == []  # 등록되지 않은 process는 건드리지 않습니다


def test_store_lock_is_exclusive(tmp_path):
    target = tmp_path / "store.json"
    with store_lock(target):
        (target.with_suffix(".json.lock")).write_text("1", encoding="utf-8")
        with pytest.raises(RunLockError, match="locked by pid"):
            with store_lock(target, timeout=0.1):
                pass
    assert not target.with_suffix(".json.lock").exists()


def test_outcome_and_label_stores_save_atomically(tmp_path):
    from aimo.labels import LabelStore, OutcomeStore, PromptOutcome

    store = OutcomeStore()
    store.merge(
        [
            PromptOutcome(
                prompt_id="a",
                counts={"C": 4},
                planned_trials=4,
                completed_trials=4,
                policy_hash="pol1",
                scorer_version="2",
            )
        ]
    )
    path = store.save(tmp_path / "outcomes.json")
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp.*"))
    assert not list(tmp_path.glob("*.lock"))
    labels = LabelStore(policy_hash="pol1", scorer_version="2")
    labels.save(tmp_path / "labels.json")
    assert (tmp_path / "labels.json").exists()
    assert not list(tmp_path.glob("*.lock"))


# --------------------------------------------------------------------------------------
# prepared registry와 parquet batch 읽기
# --------------------------------------------------------------------------------------


def write_prepared(root: pathlib.Path, n: int = 6) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    inputs = [
        {"row_id": f"r{i}", "question": f"Find the number of divisors of {100 + i} exactly."}
        for i in range(n)
    ]
    answers = [{"row_id": f"r{i}", "final_answer": str(i)} for i in range(n)]
    metadata = [
        {
            "row_id": f"r{i}",
            "difficulty": 4.0 + i,
            "topic": "Mathematics -> Number Theory -> Divisors",
        }
        for i in range(n)
    ]
    (root / "inputs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in inputs), encoding="utf-8"
    )
    (root / "answers.jsonl").write_text(
        "\n".join(json.dumps(row) for row in answers), encoding="utf-8"
    )
    (root / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metadata), encoding="utf-8"
    )
    (root / "splits.json").write_text(
        json.dumps({"train": ["r0", "r1", "r2"], "validation": ["r3"],
                    "held_out_original": ["r4", "r5"]}),
        encoding="utf-8",
    )
    (root / "pairs.jsonl").write_text("", encoding="utf-8")
    return root


def test_prepared_registry_uses_frozen_splits(tmp_path):
    from aimo.adapters.deepmath import (
        assign_splits,
        load_prepared_registry,
        select_candidates,
    )

    root = write_prepared(tmp_path / "prepared")
    bundle = load_prepared_registry(root)
    assert len(bundle["rows"]) == 6
    assert bundle["frozen_splits"]["train"] == ["r0", "r1", "r2"]
    report = select_candidates(bundle["rows"], max_originals=10)
    frozen = assign_splits(report.candidates, frozen=bundle["frozen_splits"])
    assert [row.row_id for row in frozen["train"]] == ["r0", "r1", "r2"]
    assert [row.row_id for row in frozen["validation"]] == ["r3"]
    # frozen이 없으면 새로 배정하고, frozen이 있으면 덮어쓰지 않습니다.
    fresh = assign_splits(report.candidates)
    assert [row.row_id for row in fresh["train"]] != [row.row_id for row in frozen["train"]]


def test_frozen_splits_reject_duplicate_assignment(tmp_path):
    from aimo.adapters.deepmath import apply_frozen_splits, load_prepared_registry

    bundle = load_prepared_registry(write_prepared(tmp_path / "prepared"))
    with pytest.raises(ValueError, match="more than one frozen split"):
        apply_frozen_splits(bundle["rows"], {"train": ["r0"], "validation": ["r0"]})


def test_parquet_is_read_in_batches_with_selected_columns(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow is a server extra")
    import pyarrow

    from aimo.adapters.deepmath import load_local_rows

    n = 9
    table = pyarrow.table(
        {
            "row_id": [f"r{i}" for i in range(n)],
            "question": [f"Compute the value of {i} + 1 exactly." for i in range(n)],
            "final_answer": [str(i + 1) for i in range(n)],
            "difficulty": [float(i) for i in range(n)],
            "topic": ["Mathematics -> Algebra"] * n,
            "r1_solution_1": ["a very long chain of thought " * 50] * n,
            "unused_big_column": ["x" * 100] * n,
        }
    )
    path = tmp_path / "rows.parquet"
    pq.write_table(table, path)
    rows = load_local_rows(path, revision="fixture", batch_size=2)
    assert len(rows) == n
    # r1 풀이 본문은 보관하지 않고 개수만 남깁니다.
    assert rows[0].n_r1_solutions == 1
    assert not hasattr(rows[0], "r1_solution_1")
    assert rows[0].as_metadata()["difficulty_scale"] == "deepmath_native"
