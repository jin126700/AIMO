"""공식 `are_robust` entry point와 process 전체 시간 관리자.

이 경로는 frozen forward만 합니다. gradient, intervention, fitting, 다운로드, 외부 API를
호출하지 않고 `sensitivity` 같은 offline 연구 module을 import하지 않습니다.

기본 feature는 scalar 3개입니다.

1. Original–MP prompt-end native divergence (full-vocabulary JS, 최대 view 기준)
2. all-token unexplained response-change residual의 최대값
3. 같은 residual의 RMS

residual은 native / student의 paired response 차이입니다. 대응된 endpoint 쌍 (t, t')에서

    Δy     = y_orig[t] - y_view[t']                      (native logit-contrast sketch)
    Δy_hat = (R^T D)(z_orig[g, t] - z_view[g, t'])        (student, 모든 stage g)
    residual = ||Δy - Δy_hat|| / sqrt(q)

raw latent norm을 functional instability로 취급하지 않습니다. 길이·confidence·alignment
coverage는 control feature로 함께 기록합니다.

시간 관리: 제한은 모든 model·문제를 합친 prediction run 전체 3600초이고 내부 목표는
2700초입니다. process 전체 monotonic clock으로 여러 `are_robust` 호출에 걸쳐 관리합니다.
새 forward를 시작하기 전에 비용을 예측하고 여유가 부족하면 fallback prior를 씁니다.
fallback도 모든 문제에 실제 Python bool을 돌려줍니다.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .model_registry import ADAPTER_NONE, MODEL_REGISTRY
from .mp_views import align_endpoints, make_mp_views
from .native_extract import (
    PromptPolicy,
    StreamSink,
    loaded_revision,
    pad_right,
    render_prompt,
    run_extraction,
    tokenize_prompt,
)
from .official_contract import (
    DEFAULT_EFFORT,
    INTERNAL_TARGET_SECONDS,
    SMALL_TRACK_MARKER,
    STARTER_COMMIT,
    TIME_LIMIT_SECONDS,
)
from .robust_predictor import LogisticPredictor
from .sketch import FoldedHead, fold_head, make_sketch, output_head, sketch_hash

SUBMISSION_ARTIFACT_SCHEMA = "aimo-submission-artifact-v1"
ARTIFACT_JSON = "submission_artifact.json"

# import 시점을 process 시작의 가장 이른 근사로 씁니다.
_PROCESS_START = time.monotonic()

# problem_features가 돌려주는 feature 이름.
FEATURE_NAMES = (
    "prompt_end_divergence",
    "residual_max",
    "residual_rms",
    "n_tokens",
    "prompt_end_entropy",
    "prompt_end_max_prob",
    "alignment_coverage",
    "n_views",
)

REASON_UNSUPPORTED = "unsupported_model"
REASON_NO_ARTIFACT = "no_artifact"
REASON_UNPINNED = "revision_unpinned"
REASON_REVISION = "revision_mismatch"
REASON_LOAD = "model_load_failed"
REASON_DEADLINE = "deadline"
REASON_FEATURE = "feature_error"
REASON_INTERNAL = "internal_error"


@dataclass
class CostModel:
    """forward 비용 예측. artifact의 server 실측 값이 prior이고 실행 중 EMA로 갱신합니다."""

    load_seconds: float = 60.0
    fold_seconds: float = 15.0
    seconds_per_token: float = 2e-3
    overhead_per_problem: float = 0.1
    safety: float = 1.5
    ema: float = 0.3

    def forward_seconds(self, n_tokens: int) -> float:
        return self.safety * (self.overhead_per_problem + self.seconds_per_token * n_tokens)

    def observe(self, n_tokens: int, seconds: float) -> None:
        if n_tokens > 0:
            observed = max(seconds - self.overhead_per_problem, 0.0) / n_tokens
            self.seconds_per_token = (1 - self.ema) * self.seconds_per_token + self.ema * observed


class Deadline:
    """process 전체 monotonic deadline. 호출이 여러 번이어도 같은 시작점을 씁니다."""

    def __init__(
        self,
        target: float = INTERNAL_TARGET_SECONDS,
        hard: float = TIME_LIMIT_SECONDS,
        reserve: float = 30.0,
        start: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if target > hard:
            raise ValueError("internal target must not exceed the hard limit")
        self.target, self.hard, self.reserve = target, hard, reserve
        self.clock = clock
        self.start = _PROCESS_START if start is None else start

    def elapsed(self) -> float:
        return self.clock() - self.start

    def affordable(self, seconds: float) -> bool:
        return self.elapsed() + seconds + self.reserve <= self.target


@dataclass
class ModelArtifact:
    model_id: str
    model_revision: str | None
    tokenizer_revision: str | None
    n_macro: int
    rank: int
    sketch_q: int
    sketch_seed: int
    sketch_hash: str
    prompt_policy: dict
    predictor: dict
    cost_model: dict
    tensor_file: str
    tensor_sha256: str
    stage_e_config_hash: str = ""
    validation: dict = field(default_factory=dict)


@dataclass
class SubmissionArtifact:
    models: dict[str, ModelArtifact]
    fallback_prior: dict
    schema: str = SUBMISSION_ARTIFACT_SCHEMA
    official_starter_commit: str = STARTER_COMMIT
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["models"] = {k: asdict(v) for k, v in self.models.items()}
        return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifact(directory: str | Path) -> SubmissionArtifact:
    directory = Path(directory)
    payload = json.loads((directory / ARTIFACT_JSON).read_text(encoding="utf-8"))
    if payload.get("schema") != SUBMISSION_ARTIFACT_SCHEMA:
        raise ValueError(f"artifact schema {payload.get('schema')!r} is not supported")
    models = {k: ModelArtifact(**v) for k, v in payload["models"].items()}
    for entry in models.values():
        if _sha256(directory / entry.tensor_file) != entry.tensor_sha256:
            raise ValueError(f"artifact tensor {entry.tensor_file} does not match its sha256")
    return SubmissionArtifact(models=models, fallback_prior=payload.get("fallback_prior", {}),
                              notes=payload.get("notes", []))


def write_artifact(
    directory: str | Path,
    entries: dict[str, tuple[ModelArtifact, dict[str, Tensor]]],
    fallback_prior: dict,
    notes: list[str] | None = None,
) -> Path:
    """model별 tensor(`encoder`, `offset`, `student_weight`, `student_bias`)와 JSON을 씁니다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    models = {}
    for model_id, (meta, tensors) in entries.items():
        if meta.model_revision is None:
            raise ValueError(f"{model_id}: artifact needs the pinned model revision it was fit on")
        path = directory / meta.tensor_file
        torch.save({k: v.detach().cpu().float().contiguous() for k, v in tensors.items()}, path)
        meta.tensor_sha256 = _sha256(path)
        models[model_id] = meta
    artifact = SubmissionArtifact(models=models, fallback_prior=fallback_prior, notes=notes or [])
    out = directory / ARTIFACT_JSON
    out.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out


@dataclass
class LoadedModel:
    model_id: str
    model: torch.nn.Module
    tokenizer: object
    folded: FoldedHead
    tensors: dict[str, Tensor]
    predictor: LogisticPredictor | None
    policy: PromptPolicy
    cost: CostModel


def _default_loader(model_id: str, revision: str | None, device: str):
    from .native_extract import load_frozen_model

    return load_frozen_model(model_id, revision=revision, device=device, local_files_only=True)


@dataclass
class RuntimeState:
    deadline: Deadline = field(default_factory=Deadline)
    artifact_dir: Path | None = None
    artifact: SubmissionArtifact | None = None
    artifact_error: str | None = None
    loader: Callable = _default_loader
    device: str = "cuda"
    current: LoadedModel | None = None
    stats: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    per_model: dict = field(default_factory=dict)
    last_call_fallback: list[bool] = field(default_factory=list)


_STATE = RuntimeState()


def configure_runtime(
    artifact_dir: str | Path | None = None,
    *,
    loader: Callable | None = None,
    device: str | None = None,
    deadline: Deadline | None = None,
    reset: bool = False,
) -> RuntimeState:
    """bundle의 solution.py와 test가 부릅니다. artifact를 한 번만 읽고 검증합니다."""
    global _STATE
    if reset:
        _release(_STATE)
        _STATE = RuntimeState()
    if deadline is not None:
        _STATE.deadline = deadline
    if loader is not None:
        _STATE.loader = loader
    if device is not None:
        _STATE.device = device
    elif not torch.cuda.is_available():
        _STATE.device = "cpu"
    if artifact_dir is not None:
        _STATE.artifact_dir = Path(artifact_dir)
        try:
            _STATE.artifact = load_artifact(artifact_dir)
            _STATE.artifact_error = None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _STATE.artifact, _STATE.artifact_error = None, f"{type(exc).__name__}: {exc}"
    return _STATE


def _release(state: RuntimeState) -> None:
    if state.current is not None:
        state.current = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _fallback(state: RuntimeState, model_id: str, effort: str) -> bool:
    prior = state.artifact.fallback_prior if state.artifact else {}
    for key in (f"{model_id}|{effort}", model_id, "*"):
        if key in prior:
            return bool(prior[key]["prediction"])
    return False  # 검증되지 않은 상수 fallback. 보고서에 따로 표시합니다.


def _ensure_model(state: RuntimeState, model_id: str) -> tuple[LoadedModel | None, str | None]:
    entry = MODEL_REGISTRY.get(model_id)
    if entry is None or entry.adapter_status == ADAPTER_NONE:
        return None, REASON_UNSUPPORTED
    if state.artifact is None or model_id not in state.artifact.models:
        return None, REASON_NO_ARTIFACT
    meta = state.artifact.models[model_id]
    if meta.model_revision is None:
        return None, REASON_UNPINNED
    if state.current is not None and state.current.model_id == model_id:
        return state.current, None
    _release(state)
    cost = CostModel(**meta.cost_model)
    if not state.deadline.affordable(cost.load_seconds + cost.fold_seconds):
        return None, REASON_DEADLINE
    try:
        model, tokenizer = state.loader(model_id, meta.model_revision, state.device)
    except Exception:  # noqa: BLE001 - load 실패는 fallback으로 처리합니다.
        return None, REASON_LOAD
    if loaded_revision(model) != meta.model_revision:
        return None, REASON_REVISION
    vocab = output_head(model).weight.shape[0]
    sketch = make_sketch(vocab, meta.sketch_q, meta.sketch_seed)
    if sketch_hash(sketch) != meta.sketch_hash:
        return None, REASON_REVISION
    tensors = torch.load(state.artifact_dir / meta.tensor_file, map_location="cpu",
                         weights_only=True)
    state.current = LoadedModel(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        folded=fold_head(model, sketch, meta.sketch_seed),
        tensors=tensors,
        predictor=LogisticPredictor.from_dict(meta.predictor),
        policy=PromptPolicy.from_dict(meta.prompt_policy),
        cost=cost,
    )
    return state.current, None


def _js_divergence(logp: Tensor, logq: Tensor) -> float:
    m = torch.logsumexp(torch.stack([logp, logq]), dim=0) - math.log(2.0)
    kl_pm = (logp.exp() * (logp - m)).sum()
    kl_qm = (logq.exp() * (logq - m)).sum()
    return float(0.5 * (kl_pm + kl_qm))


def prepare_problem(loaded: LoadedModel, problem: str) -> dict:
    """원문과 MP view를 tokenize합니다. forward 전 비용 예측에 씁니다."""
    views, skipped = make_mp_views(problem)
    texts = [problem] + [view.text for view in views]
    encoded = [
        tokenize_prompt(loaded.tokenizer, render_prompt(loaded.tokenizer, text, loaded.policy),
                        loaded.policy.max_tokens)
        for text in texts
    ]
    return {"views": views, "skipped": skipped, "encoded": encoded,
            "n_tokens": sum(int(e["input_ids"].shape[0]) for e in encoded)}


@torch.inference_mode()
def problem_features(loaded: LoadedModel, problem: str, prepared: dict, device: str) -> dict:
    """frozen forward 한 번(원문 + view batch)으로 feature를 계산합니다."""
    encoded = prepared["encoded"]
    pad_id = getattr(loaded.tokenizer, "pad_token_id", 0) or 0
    ids, mask = pad_right([e["input_ids"] for e in encoded], pad_id)
    lengths = mask.sum(1)
    tensors = loaded.tensors
    sink = StreamSink(tensors["encoder"].to(device), tensors["offset"].to(device), loaded.folded)
    sink.last_index = (lengths - 1).to(device)
    run_extraction(loaded.model, ids.to(device), mask.to(device), sink, tensors["encoder"].shape[0])
    z = sink.stacked()["z_state"].float()  # [B, G, T, r]
    y = sink.y.float()  # [B, T, q]
    head = output_head(loaded.model)
    logits_last = sink.final_norm_last.to(head.weight.dtype) @ head.weight.T
    if head.bias is not None:
        logits_last = logits_last + head.bias
    logp_last = torch.log_softmax(logits_last.float(), dim=-1)
    w_s = tensors["student_weight"].to(device)
    residuals, divergences, coverage = [], [], []
    for k, view in enumerate(prepared["views"], start=1):
        align = align_endpoints(
            problem, view, encoded[0]["segments"], encoded[0]["char_spans"],
            encoded[k]["segments"], encoded[k]["char_spans"],
            encoded[0]["input_ids"], encoded[k]["input_ids"],
        )
        coverage.append(align.coverage)
        divergences.append(_js_divergence(logp_last[0], logp_last[k]))
        pairs = list(align.pairs) + ([align.prompt_end] if align.prompt_end else [])
        if not pairs:
            continue
        idx_o = torch.tensor([p[0] for p in pairs], device=device)
        idx_v = torch.tensor([p[1] for p in pairs], device=device)
        dy = y[0, idx_o] - y[k, idx_v]  # [P, q]
        dz = z[0][:, idx_o] - z[k][:, idx_v]  # [G, P, r]
        dy_hat = dz @ w_s.T  # [G, P, q]
        residual = (dy.unsqueeze(0) - dy_hat).norm(dim=-1) / math.sqrt(y.shape[-1])
        residuals.append(residual.flatten())
    res = torch.cat(residuals) if residuals else torch.zeros(0)
    probs = logp_last[0].exp()
    return {
        "prompt_end_divergence": max(divergences, default=0.0),
        "residual_max": float(res.max()) if res.numel() else 0.0,
        "residual_rms": float(res.pow(2).mean().sqrt()) if res.numel() else 0.0,
        "n_tokens": float(lengths[0]),
        "prompt_end_entropy": float(-(probs * logp_last[0]).sum()),
        "prompt_end_max_prob": float(probs.max()),
        "alignment_coverage": sum(coverage) / len(coverage) if coverage else 0.0,
        "n_views": float(len(prepared["views"])),
    }


def _predict(state: RuntimeState, model_id: str, effort: str, problems: list[str]) -> list[bool]:
    n = len(problems)
    fallback_flags = [True] * n
    predictions = [_fallback(state, model_id, effort)] * n
    model_stats = state.per_model.setdefault(model_id, Counter())
    model_stats["problems"] += n
    loaded, reason = _ensure_model(state, model_id)
    if loaded is None:
        state.reasons[reason] += n
        model_stats[f"fallback:{reason}"] += n
        state.last_call_fallback = fallback_flags
        return predictions
    for i, problem in enumerate(problems):
        try:
            prepared = prepare_problem(loaded, problem)
            if not state.deadline.affordable(loaded.cost.forward_seconds(prepared["n_tokens"])):
                remaining = n - i
                state.reasons[REASON_DEADLINE] += remaining
                model_stats[f"fallback:{REASON_DEADLINE}"] += remaining
                break
            started = time.monotonic()
            features = problem_features(loaded, problem, prepared, state.device)
            loaded.cost.observe(prepared["n_tokens"], time.monotonic() - started)
            predictions[i] = loaded.predictor.predict([features])[0]
            fallback_flags[i] = False
            model_stats["shared_encoder"] += 1
        except Exception:  # noqa: BLE001 - 문제 하나의 실패는 그 문제만 fallback합니다.
            state.reasons[REASON_FEATURE] += 1
            model_stats[f"fallback:{REASON_FEATURE}"] += 1
    state.last_call_fallback = fallback_flags
    return predictions


def are_robust(
    model_id: str, reasoning_effort: str = DEFAULT_EFFORT, problems: list[str] | None = None
) -> list[bool]:
    """공식 entry point. 입력 순서와 길이를 보존하고 실제 Python bool list를 돌려줍니다."""
    state = _STATE
    problems = list(problems or [])
    effort = reasoning_effort if isinstance(reasoning_effort, str) and reasoning_effort.strip() \
        else DEFAULT_EFFORT
    state.stats["calls"] += 1
    state.stats["problems"] += len(problems)
    try:
        predictions = _predict(state, str(model_id), effort, [str(p) for p in problems])
    except Exception:  # noqa: BLE001 - 어떤 경우에도 모든 문제에 bool을 돌려줍니다.
        state.reasons[REASON_INTERNAL] += len(problems)
        predictions = [_fallback(state, str(model_id), effort)] * len(problems)
        state.last_call_fallback = [True] * len(problems)
    if len(predictions) != len(problems):
        predictions = [_fallback(state, str(model_id), effort)] * len(problems)
    return [True if value else False for value in predictions]


def runtime_report() -> dict:
    state = _STATE
    n_fallback = sum(state.reasons.values())
    total = state.stats["problems"]
    return {
        "calls": state.stats["calls"],
        "problems": total,
        "fallback": n_fallback,
        "fallback_rate": n_fallback / total if total else 0.0,
        "fallback_reasons": dict(state.reasons),
        "per_model": {k: dict(v) for k, v in state.per_model.items()},
        "elapsed_seconds": state.deadline.elapsed(),
        "internal_target_seconds": state.deadline.target,
        "hard_limit_seconds": state.deadline.hard,
        "artifact_loaded": state.artifact is not None,
        "artifact_error": state.artifact_error,
        "fallback_constant_unvalidated": (
            state.artifact is None or not state.artifact.fallback_prior),
        "note": (
            "fallback predictions are not evidence that the shared encoder works for that model"
        ),
    }


def evaluate_cases(cases: list[dict], labels: dict[str, bool]) -> dict:
    """ingestion과 같은 (model_id, effort) 묶음으로 are_robust를 부르고 공식 규칙으로 채점합니다.

    fallback 예측을 포함한 전체 accuracy와 fallback이 아닌 부분의 accuracy를 따로 냅니다.
    """
    batches: dict[tuple[str, str], list[dict]] = {}
    for case in cases:
        key = (case["model_id"], case.get("reasoning_effort", DEFAULT_EFFORT))
        batches.setdefault(key, []).append(case)
    correct = covered = native_total = native_correct = 0
    for (model_id, effort), batch in batches.items():
        problems = [case["problem"] for case in batch]
        results = are_robust(model_id, reasoning_effort=effort, problems=problems)
        valid = type(results) is list and len(results) == len(batch) and all(
            type(r) is bool for r in results)
        flags = _STATE.last_call_fallback
        for index, (case, result) in enumerate(zip(batch, results, strict=True)):
            if not valid or case["id"] not in labels:
                continue
            covered += 1
            hit = result == labels[case["id"]]
            correct += hit
            if index < len(flags) and not flags[index]:
                native_total += 1
                native_correct += hit
    total = len(labels)
    return {
        "accuracy_including_fallback": correct / total if total else math.nan,
        "coverage": covered / total if total else math.nan,
        "accuracy_non_fallback": native_correct / native_total if native_total else math.nan,
        "n_non_fallback": native_total,
        "runtime": runtime_report(),
    }


# bundle에 복사하는 module. offline 연구 module(sensitivity, stage_e 학습, npr)은 넣지 않습니다.
RUNTIME_MODULES = (
    "__init__.py",
    "official_contract.py",
    "model_registry.py",
    "native_page.py",
    "native_extract.py",
    "macro_page.py",
    "page.py",
    "mp_views.py",
    "sketch.py",
    "robust_predictor.py",
    "submission.py",
    "adapters/__init__.py",
)
FORBIDDEN_IN_BUNDLE = ("sensitivity.py", "stage_e_experiment.py", "npr.py", "native_data.py")

SOLUTION_TEMPLATE = '''"""AIMO Interpretability Challenge submission entry (generated)."""

from pathlib import Path

from aimo_submission.submission import are_robust as _are_robust
from aimo_submission.submission import configure_runtime

configure_runtime(artifact_dir=Path(__file__).resolve().parent / "artifacts")


def are_robust(model_id: str, reasoning_effort: str = "default", problems=None) -> list:
    return _are_robust(model_id, reasoning_effort=reasoning_effort, problems=problems)
'''


def build_bundle(out_dir: str | Path, artifact_dir: str | Path | None, *, small: bool = True,
                 make_zip: bool = False) -> Path:
    """제출 bundle directory를 만듭니다. dataset, label, model weight, cache는 넣지 않습니다."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"bundle directory {out} is not empty")
    package = out / "aimo_submission"
    source_root = Path(__file__).resolve().parent
    for name in RUNTIME_MODULES:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, target)
    (out / "solution.py").write_text(SOLUTION_TEMPLATE, encoding="utf-8")
    if small:
        (out / SMALL_TRACK_MARKER).write_text("small\n", encoding="utf-8")
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    if artifact_dir is not None:
        for path in sorted(Path(artifact_dir).iterdir()):
            if path.suffix not in (".json", ".pt"):
                raise ValueError(f"unexpected artifact file {path.name}")
            shutil.copyfile(path, artifacts / path.name)
    for name in FORBIDDEN_IN_BUNDLE:
        if (package / name).exists():
            raise RuntimeError(f"offline module {name} must not be in the bundle")
    if make_zip:
        return Path(shutil.make_archive(str(out), "zip", root_dir=out))
    return out
