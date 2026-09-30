"""Stage-E backend CLI (`native-extract`, `stage-e`, `submission-*`, `official-contract`).

legacy Looped / LRT / Flow 명령은 그대로 두고, 새 backend는 이 명령들로만 선택합니다.
실제 model이 필요한 단계는 `--execute-gpu`와 CUDA가 있어야 하며 조용히 CPU로 내려가지
않습니다. 로컬에서는 `--toy` 경로와 dry-run만 실행합니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from . import official_contract
from .config import Config, StageEConfig, load_config
from .native_page import STATUS_DATA_UNAVAILABLE

EXIT_BLOCKED = 2
EXIT_DATA_UNAVAILABLE = 3

UNVERIFIED_ITEMS = (
    "actual Qwen3.5-4B hook placement (linear_attn / self_attn / post_attention_layernorm)",
    "actual-model backward through the hybrid Mixer (VJP)",
    "inference vs gradient backend numerical agreement on the real model",
    "bf16 residual identity tolerance on the real model",
    "real folded-head identity R^T log_softmax(l) = folded(N(h))",
    "GPU VRAM for teacher VJP batches",
    "full 1-hour submission benchmark on the official runtime",
    "adapters for Skywork-OR1-Math-7B, Olmo-3-7B-Think, DeepSeek-R1-0528-Qwen3-8B",
)


def _stage_cfg(args) -> StageEConfig:
    cfg = load_config(args.config).stage_e if getattr(args, "config", None) else Config().stage_e
    if getattr(args, "n_macro", None):
        cfg = replace(cfg, n_macro=args.n_macro)
    return cfg


def _require_gpu(args) -> None:
    if not getattr(args, "execute_gpu", False):
        raise SystemExit(
            "error: this stage loads the real target model; pass --execute-gpu on the server")
    if not torch.cuda.is_available():
        raise SystemExit("error: --execute-gpu was given but CUDA is not available")


def _data_unavailable(message: str) -> int:
    print(json.dumps({"status": STATUS_DATA_UNAVAILABLE, "reason": message}, indent=2))
    return EXIT_DATA_UNAVAILABLE


def official_contract_command(_args) -> int:
    payload = {k: getattr(official_contract, k) for k in dir(official_contract) if k.isupper()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def spec_payload(cfg: StageEConfig) -> dict:
    from .model_registry import MODEL_REGISTRY

    return {
        "backend": cfg.backend,
        "stage_e": asdict(cfg),
        "registry": {k: asdict(v) for k, v in MODEL_REGISTRY.items()},
        "implemented_and_cpu_toy_tested": [
            "native all-token macro extractor (pre_norm hybrid + post_norm toy layouts)",
            "Stage-E encoder / shared linear decoder / projector / pseudoinverse",
            "coarse top-k + OTHER NPR loss with chunked normalizers, full-vocab KL audit",
            "VJP sensitivity loss", "logit-contrast sketch folding", "MP edit-map alignment",
            "are_robust runtime with process deadline and fallback",
        ],
        "unverified": list(UNVERIFIED_ITEMS),
    }


def stage_e_command(args) -> int:
    cfg = _stage_cfg(args)
    run_dir = Path(args.run_dir)
    if args.stage == "spec":
        print(json.dumps(spec_payload(cfg), indent=2, sort_keys=True, default=str))
        return 0
    if args.stage == "toy":
        from .stage_e_experiment import run_toy_experiment

        report = run_toy_experiment(cfg, run_dir, epochs=args.epochs, seed=args.seed)
        print(json.dumps({"status": report["status"], "selection": report["selection"],
                          "report": str(run_dir / "stage_e_report.json")}, indent=2, default=str,
                         ensure_ascii=False))
        return 0
    return _run_real_discovery(cfg, run_dir, args)


def _pinned(cfg: StageEConfig) -> str | None:
    if not cfg.model_revision or not cfg.tokenizer_revision:
        return "stage_e.model_revision and stage_e.tokenizer_revision must be pinned"
    return None


def _load_discovery(cfg: StageEConfig, dataset: str, seed: int):
    from .native_data import (
        ProtectedRegistry,
        assign_splits,
        load_deepmath_discovery,
        load_gsm8k_control,
    )

    if not cfg.protected_registry or not Path(cfg.protected_registry).exists():
        raise FileNotFoundError("stage_e.protected_registry (locked / known-test IDs) is required")
    protected = ProtectedRegistry.load(cfg.protected_registry)
    path = cfg.discovery_data if dataset == "deepmath" else cfg.gsm8k_data
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"{dataset} data file is not available locally: {path}")
    if dataset == "deepmath":
        problems, report = load_deepmath_discovery(path, protected=protected, seed=seed)
    else:
        problems, report = load_gsm8k_control(path, protected=protected)
    return assign_splits(problems, seed=seed), report


def _run_real_discovery(cfg: StageEConfig, run_dir: Path, args) -> int:
    from .native_extract import PromptPolicy, load_frozen_model, loaded_revision
    from .stage_e_experiment import NativeTeacher, run_discovery

    blocked = _pinned(cfg)
    if blocked:
        print(f"error: {blocked}", file=sys.stderr)
        return EXIT_BLOCKED
    try:
        problems, data_report = _load_discovery(cfg, args.dataset, args.seed)
    except FileNotFoundError as exc:
        return _data_unavailable(str(exc))
    _require_gpu(args)
    model, tokenizer = load_frozen_model(cfg.model_id, revision=cfg.model_revision, dtype=cfg.dtype,
                                         device="cuda")
    if loaded_revision(model) != cfg.model_revision:
        print("error: loaded model revision differs from stage_e.model_revision", file=sys.stderr)
        return EXIT_BLOCKED
    teacher = NativeTeacher(model, tokenizer, cfg, PromptPolicy(max_tokens=cfg.max_tokens),
                            device="cuda", cache=False)
    by_split = {name: [p for p in problems if p.split == name]
                for name in ("train", "dev", "held_out")}
    run_dir.mkdir(parents=True, exist_ok=True)
    report = run_discovery(
        teacher, by_split, cfg, run_dir, epochs=args.epochs, seed=args.seed,
        batch_size=args.batch_size,
        native_meta={"model_id": cfg.model_id, "model_revision": cfg.model_revision,
                     "tokenizer_revision": cfg.tokenizer_revision,
                     "macro_boundaries": teacher.bounds},
        dataset_meta={"dataset": args.dataset, "report": data_report,
                      "n_by_split": {k: len(v) for k, v in by_split.items()}},
        status="SERVER_RUN / REVIEW_REQUIRED",
    )
    print(json.dumps({"status": report["status"], "selection": report["selection"]}, indent=2))
    return 0


def native_extract_command(args) -> int:
    from .native_extract import PromptPolicy, extract_native_pages, model_provenance
    from .native_page import save_native_pages

    cfg = _stage_cfg(args)
    run_dir = Path(args.run_dir)
    policy = PromptPolicy(max_tokens=cfg.max_tokens)
    if args.toy:
        from .native_toy import ToyTokenizer, build_toy_hybrid
        from .stage_e_experiment import toy_problems

        model, tokenizer = build_toy_hybrid(), ToyTokenizer()
        provenance = model_provenance(model, tokenizer, cfg.n_macro, policy, model_id="aimo/toy",
                                      model_revision="toy", tokenizer_revision="toy",
                                      dataset={"source": "aimo/toy"})
        records = [dict(p.record(), split="train") for p in toy_problems()[: args.limit]]
        device = "cpu"
    else:
        from .native_extract import load_frozen_model, loaded_revision

        blocked = _pinned(cfg)
        if blocked:
            print(f"error: {blocked}", file=sys.stderr)
            return EXIT_BLOCKED
        try:
            problems, _ = _load_discovery(cfg, args.dataset, args.seed)
        except FileNotFoundError as exc:
            return _data_unavailable(str(exc))
        _require_gpu(args)
        model, tokenizer = load_frozen_model(cfg.model_id, revision=cfg.model_revision,
                                             dtype=cfg.dtype, device="cuda")
        if loaded_revision(model) != cfg.model_revision:
            print("error: loaded model revision differs from stage_e.model_revision",
                  file=sys.stderr)
            return EXIT_BLOCKED
        provenance = model_provenance(
            model, tokenizer, cfg.n_macro, policy, model_id=cfg.model_id,
            model_revision=cfg.model_revision, tokenizer_revision=cfg.tokenizer_revision,
            dataset={"dataset": args.dataset})
        records = [p.record() for p in problems[: args.limit]]
        device = "cuda"
    pages, summary = extract_native_pages(model, tokenizer, records, provenance, policy,
                                          device=device)
    save_native_pages(pages, run_dir / "native_pages")
    (run_dir / "native_extract_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def submission_bundle_command(args) -> int:
    from .submission import build_bundle

    out = build_bundle(args.out, args.artifact, small=not args.main, make_zip=args.zip)
    print(json.dumps({"bundle": str(out), "small_track": not args.main}, indent=2))
    return 0


def submission_fit_command(args) -> int:
    """feature row(JSONL)로 root-grouped predictor와 fallback prior를 고릅니다."""
    from .robust_predictor import fit_prior, select_predictor

    rows = [json.loads(line) for line in Path(args.features).read_text().splitlines() if line]
    labeled = [row for row in rows if row.get("is_robust") is not None]
    predictor, report = select_predictor(
        labeled, [bool(r["is_robust"]) for r in labeled], [r["root_id"] for r in labeled],
        feature_set=args.feature_set, seed=args.seed)
    report["n_null_label_excluded"] = len(rows) - len(labeled)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictor.json").write_text(json.dumps(predictor.to_dict(), indent=2))
    (out / "fallback_prior.json").write_text(
        json.dumps(fit_prior(labeled, [bool(r["is_robust"]) for r in labeled]), indent=2))
    (out / "selection_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["selected"], indent=2))
    return 0


def submission_local_command(args) -> int:
    """공식 cases.jsonl / labels.jsonl 형식으로 bundle 없이 are_robust를 돌려 봅니다."""
    from .submission import configure_runtime, evaluate_cases

    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line]
    labels = {row["id"]: row["is_robust"] for row in
              (json.loads(line) for line in Path(args.labels).read_text().splitlines() if line)}
    configure_runtime(args.artifact, device=args.device, reset=True)
    print(json.dumps(evaluate_cases(cases, labels), indent=2, default=str))
    return 0


def artifact_tensors(stage_e_model, sketch: torch.Tensor) -> dict[str, torch.Tensor]:
    """frozen Stage-E를 제출 runtime tensor로 바꿉니다 (encoder, offset, R^T D, R^T b)."""
    with torch.no_grad():
        weight, bias = stage_e_model.folded_decoder(sketch)
        return {"encoder": stage_e_model.encoder_matrix(), "offset": stage_e_model.offset.detach(),
                "student_weight": weight, "student_bias": bias}


def feature_rows(model, tokenizer, stage_e_model, meta: dict, label_rows: list, *,
                 model_id: str, device: str) -> list[dict]:
    """공식 train label row마다 frozen feature를 계산합니다. 같은 문제는 한 번만 계산합니다.

    공간(encoder, rank)과 feature 정의는 이미 freeze된 상태여야 합니다. label은 predictor
    학습에만 쓰고 feature 계산에는 들어가지 않습니다.
    """
    from .native_extract import PromptPolicy
    from .sketch import fold_head, make_sketch, output_head
    from .submission import CostModel, LoadedModel, prepare_problem, problem_features

    sketch_cfg = meta["sketch"]
    vocab = output_head(model).weight.shape[0]
    sketch = make_sketch(vocab, sketch_cfg["q"], sketch_cfg["seed"])
    loaded = LoadedModel(
        model_id=model_id, model=model, tokenizer=tokenizer,
        folded=fold_head(model, sketch, sketch_cfg["seed"]),
        tensors=artifact_tensors(stage_e_model, sketch), predictor=None,
        policy=PromptPolicy.from_dict(meta.get("prompt_policy", {})), cost=CostModel(),
    )
    cache: dict[str, dict] = {}
    rows = []
    for row in label_rows:
        if row.model_id != model_id:
            continue
        if row.problem not in cache:
            cache[row.problem] = problem_features(loaded, row.problem,
                                                  prepare_problem(loaded, row.problem), device)
        rows.append({**cache[row.problem], "model_id": row.model_id,
                     "reasoning_effort": row.reasoning_effort, "root_id": row.root_id,
                     "split": "train", "is_robust": row.is_robust})
    return rows


def submission_features_command(args) -> int:
    from .native_data import load_official_labels
    from .native_extract import load_frozen_model, loaded_revision
    from .stage_e import load_stage_e

    cfg = _stage_cfg(args)
    blocked = _pinned(cfg)
    if blocked:
        print(f"error: {blocked}", file=sys.stderr)
        return EXIT_BLOCKED
    if not Path(args.labels).exists():
        return _data_unavailable(f"official label file {args.labels} is not available locally")
    rows, summary = load_official_labels(args.labels)
    stage_e_model, meta = load_stage_e(args.stage_e)
    if meta["native"].get("model_revision") != cfg.model_revision:
        print("error: Stage-E checkpoint was fit on a different model revision", file=sys.stderr)
        return EXIT_BLOCKED
    _require_gpu(args)
    model, tokenizer = load_frozen_model(cfg.model_id, revision=cfg.model_revision,
                                         dtype=cfg.dtype, device="cuda")
    if loaded_revision(model) != cfg.model_revision:
        print("error: loaded model revision differs from stage_e.model_revision", file=sys.stderr)
        return EXIT_BLOCKED
    out = feature_rows(model, tokenizer, stage_e_model.cuda(), meta, rows,
                       model_id=cfg.model_id, device="cuda")
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"label_summary": summary, "n_feature_rows": len(out)}, indent=2))
    return 0


def submission_artifact_command(args) -> int:
    from .sketch import make_sketch, sketch_hash
    from .stage_e import load_stage_e
    from .submission import ModelArtifact, write_artifact

    cfg = _stage_cfg(args)
    stage_e_model, meta = load_stage_e(args.stage_e)
    fit_dir = Path(args.fit)
    predictor = json.loads((fit_dir / "predictor.json").read_text())
    prior = json.loads((fit_dir / "fallback_prior.json").read_text())
    sketch = make_sketch(meta["vocab_size"], meta["sketch"]["q"], meta["sketch"]["seed"])
    cost = json.loads(Path(args.cost).read_text()) if args.cost else {}
    entry = ModelArtifact(
        model_id=cfg.model_id, model_revision=meta["native"].get("model_revision"),
        tokenizer_revision=meta["native"].get("tokenizer_revision"), n_macro=meta["n_macro"],
        rank=meta["rank"], sketch_q=meta["sketch"]["q"], sketch_seed=meta["sketch"]["seed"],
        sketch_hash=sketch_hash(sketch), prompt_policy=meta.get("prompt_policy", {}),
        predictor=predictor, cost_model=cost, tensor_file=f"{cfg.model_id.replace('/', '__')}.pt",
        tensor_sha256="", stage_e_config_hash=meta.get("config_hash", ""),
    )
    tensors = artifact_tensors(stage_e_model, sketch)
    path = write_artifact(args.out, {cfg.model_id: (entry, tensors)}, prior)
    print(json.dumps({"artifact": str(path)}, indent=2))
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("stage-e", help="Stage-E native representation discovery (new backend)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--stage", choices=["spec", "toy", "discovery"], required=True)
    p.add_argument("--dataset", choices=["deepmath", "gsm8k"], default="deepmath")
    p.add_argument("--n-macro", type=int, default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--execute-gpu", action="store_true")
    p.set_defaults(func=stage_e_command)

    p = sub.add_parser("native-extract", help="macro x all-token native page audit extraction")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--toy", action="store_true", help="CPU toy model")
    p.add_argument("--dataset", choices=["deepmath", "gsm8k"], default="deepmath")
    p.add_argument("--n-macro", type=int, default=None)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--execute-gpu", action="store_true")
    p.set_defaults(func=native_extract_command)

    p = sub.add_parser("submission-fit", help="root-grouped robustness predictor selection")
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--feature-set", default="core3")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=submission_fit_command)

    p = sub.add_parser("submission-features", help="frozen features for official train labels")
    p.add_argument("--stage-e", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--execute-gpu", action="store_true")
    p.set_defaults(func=submission_features_command)

    p = sub.add_parser("submission-artifact", help="combine frozen Stage-E and predictor")
    p.add_argument("--stage-e", required=True)
    p.add_argument("--fit", required=True, help="directory from submission-fit")
    p.add_argument("--out", required=True)
    p.add_argument("--cost", default=None, help="server-measured CostModel JSON")
    p.add_argument("--config", default=None)
    p.set_defaults(func=submission_artifact_command)

    p = sub.add_parser("submission-bundle", help="build a Codabench bundle directory")
    p.add_argument("--out", required=True)
    p.add_argument("--artifact", default=None)
    p.add_argument("--main", action="store_true", help="main track (no small.txt)")
    p.add_argument("--zip", action="store_true")
    p.set_defaults(func=submission_bundle_command)

    p = sub.add_parser("submission-local", help="run are_robust on official-format cases")
    p.add_argument("--cases", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--artifact", default=None)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=submission_local_command)

    p = sub.add_parser("official-contract", help="print the recorded official contract")
    p.set_defaults(func=official_contract_command)
