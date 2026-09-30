"""are_robust runtime: 반환 계약, process deadline, fallback, toy end-to-end, bundle."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest
import torch

from aimo import submission
from aimo.model_registry import MODEL_REGISTRY
from aimo.native_data import OfficialLabelRow
from aimo.native_toy import ToyTokenizer, build_toy_hybrid
from aimo.official_contract import SMALL_TRACK_MODELS
from aimo.robust_predictor import fit_prior, select_predictor
from aimo.sketch import make_sketch, sketch_hash
from aimo.stage_e import StageE
from aimo.stage_e_cli import artifact_tensors, feature_rows

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "toy-revision"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _toy_loader(revision=REVISION):
    def load(model_id, _revision, _device):
        model = build_toy_hybrid(n_layers=8)
        model.config._commit_hash = revision
        return model, ToyTokenizer()

    return load


PROBLEMS = [
    "Let x  be real.\nIf 3x + 4 = 10,\nfind  x.",
    "If  n > 2 and n = 5,\ncompute n^2 - 1.",
    "Find  y if 2y = 8.",
    "Let a  be 3.\nCompute a + 4.",
]


def _artifact(tmp_path, labels=(True, False, True, False)):
    """toy target에서 feature를 뽑아 predictor를 고르고 artifact를 씁니다 (plumbing 검증 전용)."""
    model, tok = _toy_loader()(MODEL_ID, REVISION, "cpu")
    stage = StageE(n_macro=4, hidden_size=16, rank=4, vocab_size=48, seed=0)
    meta = {"sketch": {"q": 8, "seed": 11}, "prompt_policy": {"max_tokens": 512}}
    rows = [OfficialLabelRow(MODEL_ID, "d", str(i), text, "default", label, root_id=f"r{i}")
            for i, (text, label) in enumerate(zip(PROBLEMS * 2, labels * 2, strict=True))]
    feats = feature_rows(model, tok, stage, meta, rows, model_id=MODEL_ID, device="cpu")
    for row in feats:
        assert set(submission.FEATURE_NAMES) <= set(row)
    predictor, _ = select_predictor(feats, [r["is_robust"] for r in feats],
                                    [r["root_id"] for r in feats], l2_grid=(1.0,), k=2)
    sketch = make_sketch(48, 8, 11)
    entry = submission.ModelArtifact(
        model_id=MODEL_ID, model_revision=REVISION, tokenizer_revision=REVISION, n_macro=4,
        rank=4, sketch_q=8, sketch_seed=11, sketch_hash=sketch_hash(sketch),
        prompt_policy={"max_tokens": 512}, predictor=predictor.to_dict(),
        cost_model={"load_seconds": 1.0, "fold_seconds": 1.0, "seconds_per_token": 1e-3},
        tensor_file="qwen.pt", tensor_sha256="",
    )
    prior = fit_prior([{"model_id": MODEL_ID, "reasoning_effort": "default"}] * 3, [True] * 3)
    submission.write_artifact(tmp_path, {MODEL_ID: (entry, artifact_tensors(stage, sketch))}, prior)
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_runtime():
    submission.configure_runtime(reset=True, device="cpu")
    yield
    submission.configure_runtime(reset=True, device="cpu")


def test_registry_matches_official_small_track():
    assert tuple(k for k, v in MODEL_REGISTRY.items() if v.small_track) == SMALL_TRACK_MODELS
    assert len(SMALL_TRACK_MODELS) == 4
    assert all(MODEL_REGISTRY[m].adapter_status for m in SMALL_TRACK_MODELS)


def test_signature_matches_ingestion_keyword_call():
    params = inspect.signature(submission.are_robust).parameters
    assert list(params)[:3] == ["model_id", "reasoning_effort", "problems"]


def test_fallback_without_artifact_returns_python_bools_in_order():
    out = submission.are_robust(MODEL_ID, reasoning_effort="low", problems=["a", "b", "c"])
    assert out == [False, False, False] and all(type(x) is bool for x in out)
    assert submission.are_robust("unknown/model", problems=[]) == []
    report = submission.runtime_report()
    assert report["fallback"] == 3 and report["fallback_constant_unvalidated"]


def test_end_to_end_toy_runtime_uses_shared_encoder(tmp_path):
    submission.configure_runtime(_artifact(tmp_path), loader=_toy_loader(), device="cpu")
    out = submission.are_robust(MODEL_ID, reasoning_effort="default", problems=PROBLEMS)
    assert len(out) == len(PROBLEMS) and all(type(x) is bool for x in out)
    assert submission.runtime_report()["per_model"][MODEL_ID]["shared_encoder"] == len(PROBLEMS)
    # 같은 model의 다른 effort 호출은 load한 model을 재사용합니다.
    loaded = submission._STATE.current
    submission.are_robust(MODEL_ID, reasoning_effort="low", problems=PROBLEMS[:1])
    assert submission._STATE.current is loaded


def test_revision_mismatch_falls_back_to_prior(tmp_path):
    submission.configure_runtime(_artifact(tmp_path), loader=_toy_loader("other"), device="cpu")
    out = submission.are_robust(MODEL_ID, problems=PROBLEMS)
    assert out == [True] * len(PROBLEMS)  # prior 다수결
    assert submission.runtime_report()["fallback_reasons"] == {"revision_mismatch": 4}


def test_unsupported_models_fall_back_without_claiming_encoder(tmp_path):
    submission.configure_runtime(_artifact(tmp_path), loader=_toy_loader(), device="cpu")
    out = submission.are_robust("allenai/Olmo-3-7B-Think", problems=PROBLEMS[:2])
    out += submission.are_robust("openai/gpt-oss-120b", problems=PROBLEMS[:1])
    assert len(out) == 3 and all(type(x) is bool for x in out)
    report = submission.runtime_report()
    assert report["fallback_reasons"] == {"no_artifact": 2, "unsupported_model": 1}
    assert "shared_encoder" not in report["per_model"]["allenai/Olmo-3-7B-Think"]


def test_process_deadline_spans_calls_and_falls_back(tmp_path):
    clock = FakeClock()
    deadline = submission.Deadline(target=100.0, hard=3600.0, reserve=0.0, start=0.0, clock=clock)
    submission.configure_runtime(_artifact(tmp_path), loader=_toy_loader(), device="cpu",
                                 deadline=deadline)
    first = submission.are_robust(MODEL_ID, problems=PROBLEMS[:2])
    assert submission.runtime_report()["fallback"] == 0
    clock.now = 99.9  # 다음 호출은 같은 process deadline을 봅니다.
    second = submission.are_robust(MODEL_ID, reasoning_effort="low", problems=PROBLEMS)
    report = submission.runtime_report()
    assert report["fallback_reasons"] == {"deadline": len(PROBLEMS)}
    assert second == [True] * len(PROBLEMS)
    assert len(first) == 2 and all(type(x) is bool for x in first + second)


def test_cost_prediction_stops_before_exceeding_budget(tmp_path):
    clock = FakeClock()
    deadline = submission.Deadline(target=60.0, reserve=0.0, start=0.0, clock=clock)
    submission.configure_runtime(_artifact(tmp_path), loader=_toy_loader(), device="cpu",
                                 deadline=deadline)
    submission._ensure_model(submission._STATE, MODEL_ID)
    submission._STATE.current.cost.seconds_per_token = 1.0  # 문제 하나가 ~100초로 예측됨
    out = submission.are_robust(MODEL_ID, problems=PROBLEMS)
    assert len(out) == 4 and submission.runtime_report()["fallback_reasons"] == {"deadline": 4}


def test_evaluate_cases_reports_fallback_and_non_fallback_accuracy(tmp_path):
    submission.configure_runtime(_artifact(tmp_path), loader=_toy_loader(), device="cpu")
    cases = [{"id": f"c{i}", "model_id": MODEL_ID, "problem": p} for i, p in enumerate(PROBLEMS)]
    cases.append({"id": "c9", "model_id": "Skywork/Skywork-OR1-Math-7B", "problem": "x"})
    labels = {c["id"]: True for c in cases}
    result = submission.evaluate_cases(cases, labels)
    assert result["coverage"] == 1.0
    assert result["n_non_fallback"] == len(PROBLEMS)
    assert result["runtime"]["fallback"] == 1


def test_bundle_has_small_marker_and_no_offline_modules(tmp_path):
    artifact = _artifact(tmp_path / "artifact")
    bundle = submission.build_bundle(tmp_path / "bundle", artifact, small=True)
    names = {p.name for p in (bundle / "aimo_submission").rglob("*.py")}
    assert (bundle / "small.txt").exists() and (bundle / "solution.py").exists()
    assert not names & {"sensitivity.py", "stage_e_experiment.py", "npr.py", "native_data.py"}
    code = (
        "import sys, inspect; sys.path.insert(0, sys.argv[1]);"
        "import importlib.util as u;"
        "spec = u.spec_from_file_location('participant_solution', sys.argv[1] + '/solution.py');"
        "m = u.module_from_spec(spec); spec.loader.exec_module(m);"
        "assert 'reasoning_effort' in inspect.signature(m.are_robust).parameters;"
        "assert not any('sensitivity' in k for k in sys.modules);"
        "r = m.are_robust('Qwen/Qwen3.5-4B', reasoning_effort='low', problems=['a', 'b']);"
        "assert type(r) is list and all(type(x) is bool for x in r) and len(r) == 2;"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code, str(bundle)], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_submission_runtime_does_not_import_offline_modules():
    code = ("import sys; import aimo.submission;"
            "bad = [m for m in ('aimo.sensitivity', 'aimo.stage_e_experiment', 'aimo.npr')"
            " if m in sys.modules]; print(bad)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.strip() == "[]", result.stderr


def test_frozen_forward_only():
    assert torch.is_inference_mode_enabled() is False
    src = inspect.getsource(submission)
    for forbidden in ("backward(", "autograd.grad", ".step(", "hf_hub_download", "requests."):
        assert forbidden not in src
