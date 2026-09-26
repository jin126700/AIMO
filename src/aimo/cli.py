"""aimo CLI. python -m aimo 도 같은 진입점을 씁니다.

명령
  check            환경과 데이터 계약 self-check
  make-toy         synthetic Page + behavior label 생성
  prepare-data     DeepMath 후보/split 준비 (legacy: MathGAP)
  import-pairs     검증된 original-variant pair import 후 freeze
  collect-outcomes 행동 측정 결과(outcome counts) 적재 (실제 generation은 --execute-gpu)
  build-labels     outcome counts -> pair drop / panel label store
  extract          Page 추출 (--tiny는 CPU random-init 검증, 실제 weights는 --execute-gpu)
  train            behavior / flow / joint 학습
  resume           같은 config로 train 이어서 실행
  evaluate         checkpoint 평가(behavior + flow), 또는 여러 run 비교
  predict          checkpoint로 한 batch V_hat 예측 (flow view)
  predict-behavior checkpoint로 pair drop / panel robust probability 예측
  preflight        서버 실행 전 dry-run (model weights를 읽지 않습니다)
  screen           legacy screening (--execute-gpu 필요)
  run-stage1       legacy flow 경로 단계 실행

기본 동작은 dry-run/preflight이며 model weights를 로드하지 않습니다. 실제 GPU 실행은
--execute-gpu를 명시해야 합니다.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from .config import Config, load_config
from .data import (
    attach_labels,
    collate,
    collate_panels,
    compute_norm_stats,
    group_by_cut,
    load_dataset,
    make_synthetic_dataset,
    sample_pairs,
    save_dataset,
)
from .evaluate import evaluate_behavior, evaluate_dataset
from .labels import BinaryRobustPolicy, LabelStore, OutcomeStore, build_label_store
from .model import build_model
from .runtime import (
    BudgetGuard,
    DedupLedger,
    GpuBudget,
    GpuBudgetExceeded,
    atomic_write_json,
    guard_config,
    run_lock,
)
from .train import checkpoint_info, load_checkpoint, train

DEFAULT_CONFIG = "configs/toy.yaml"


class CliError(RuntimeError):
    """사용자가 고칠 수 있는 오류. traceback 없이 메시지만 보여줍니다."""


# --------------------------------------------------------------------------------------
# 공통 helper
# --------------------------------------------------------------------------------------


def _parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise CliError(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        out[key.strip()] = yaml_scalar(raw.strip())
    return out


def yaml_scalar(raw: str) -> Any:
    """--set 값의 최소 타입 변환."""
    import yaml

    return yaml.safe_load(raw)


def _load(args: argparse.Namespace) -> Config:
    path = Path(args.config)
    if not path.exists():
        raise CliError(f"config not found: {path}")
    overrides = _parse_overrides(args.set)
    if getattr(args, "run_id", None):
        overrides["run.run_id"] = args.run_id
    cfg = load_config(path, overrides)
    if cfg.run.device != "cpu" and not getattr(args, "execute_gpu", False):
        raise CliError(
            f"config asks for device {cfg.run.device!r}; "
            "pass --execute-gpu to run on an accelerator"
        )
    return cfg


def _datasets(cfg: Config):
    if cfg.data.source == "synthetic":
        return make_synthetic_dataset(cfg)
    if cfg.data.source in ("pages", "deepmath"):
        if not cfg.data.page_dir:
            raise CliError(f"data.source={cfg.data.source} requires data.page_dir")
        page_dir = Path(cfg.data.page_dir)
        if not (page_dir / "index.json").exists():
            raise CliError(
                f"no extracted pages at {page_dir} (missing index.json); "
                "run 'aimo extract' first, or point data.page_dir at an existing page directory"
            )
        datasets = load_dataset(page_dir)
        if cfg.data.label_path:
            label_path = Path(cfg.data.label_path)
            if not label_path.exists():
                raise CliError(
                    f"label store not found: {label_path}; run 'aimo build-labels' first"
                )
            attach_labels(datasets, LabelStore.load(label_path))
        return datasets
    raise CliError(f"unknown data.source {cfg.data.source!r}")


def _budget_gate(cfg: Config) -> str:
    """GPU 작업을 시작해도 되는지 확인합니다. 누적 예산을 넘었으면 막습니다."""
    budget = GpuBudget.open(
        cfg.run_dir,
        block_minutes=cfg.server.gpu_block_minutes,
        stop_minutes=cfg.server.gpu_stop_minutes,
    )
    can_start, reason = budget.can_start()
    if not can_start:
        raise CliError(reason)
    return reason


def _emit(payload: dict) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


# --------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    from .adapters.deepmath import probe_deepmath
    from .adapters.mathgap import probe_mathgap
    from .adapters.qwen import probe_qwen, thinking_ready
    from .data import DATA_SCHEMA_VERSION
    from .page import Page
    from .train import CHECKPOINT_SCHEMA_VERSION, _label_coverage

    cfg = _load(args)
    datasets = _datasets(cfg)
    train_set = datasets["train"]
    for group in train_set.groups[:2]:
        group.original.validate()
        for variant in group.variants:
            variant.validate()
    stats = compute_norm_stats(train_set, floor=cfg.data.scale_floor)
    model = build_model(cfg, train_set.hidden_size, train_set.n_blocks, train_set.n_landmarks)
    report = model.param_report()
    payload = {
        "status": "ok",
        "aimo_version": __import__("aimo").__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": cfg.run.device,
        "config_hash": cfg.hash(),
        "page_contract": {
            "state_shape": list(train_set.groups[0].original.state.shape),
            "updates_shape": list(train_set.groups[0].original.updates.shape),
            "streams": ["mixer", "ffn"],
            "residual_identity_max_error": train_set.groups[0].original.residual_identity_error(),
            "page_class": Page.__name__,
        },
        "model": {"name": cfg.model.name, "params": report.as_dict()},
        "norm_stats": {
            "source": stats.source,
            "hash": stats.hash(),
            "inactive_target_cells": int((~stats.target_active).sum()),
        },
        "adapters": {
            "mathgap": probe_mathgap(),
            "qwen": probe_qwen(),
            "deepmath": probe_deepmath(),
        },
        "task": cfg.train.task,
        "select_metric": cfg.train.resolved_select_metric(),
        "schema": {
            "page_store": DATA_SCHEMA_VERSION,
            "checkpoint": CHECKPOINT_SCHEMA_VERSION,
        },
        "label_coverage": _label_coverage(train_set),
        "thinking_profile": thinking_ready(cfg.server.thinking),
    }
    return _emit(payload)


# --------------------------------------------------------------------------------------
# make-toy
# --------------------------------------------------------------------------------------


def cmd_make_toy(args: argparse.Namespace) -> int:
    cfg = _load(args)
    if cfg.data.source != "synthetic":
        raise CliError("make-toy requires data.source=synthetic")
    datasets = make_synthetic_dataset(cfg)
    out_dir = Path(args.out or (cfg.run_dir / "toy_pages"))
    save_dataset(datasets, out_dir)
    payload = {
        "status": "ok",
        "out_dir": str(out_dir),
        "splits": {
            name: {
                "n_originals": len(dataset.groups),
                "n_variants": sum(len(g.variants) for g in dataset.groups),
                "n_identity": sum(1 for g in dataset.groups for v in g.variants if v.is_identity),
                "data_hash": dataset.data_hash(),
            }
            for name, dataset in datasets.items()
        },
        "shapes": {
            "state": list(datasets["train"].groups[0].original.state.shape),
            "updates": list(datasets["train"].groups[0].original.updates.shape),
        },
        "note": "E0 검증용 synthetic Page입니다. 실제 robust dataset이 아닙니다.",
    }
    return _emit(payload)


# --------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _load(args)
    datasets = _datasets(cfg)
    with run_lock(cfg.run_dir):
        summary = train(cfg, datasets, resume=args.resume)
    summary["history"] = summary["history"][-3:]  # 출력은 마지막 몇 epoch만 보여줍니다.
    return _emit({"status": "ok", **summary})


# --------------------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.compare:
        return _compare_runs(args.compare, args.out)
    cfg = _load(args)
    datasets = _datasets(cfg)
    ckpt = Path(args.checkpoint) if args.checkpoint else cfg.run_dir / "best.pt"
    if not ckpt.exists():
        raise CliError(f"checkpoint not found: {ckpt}")
    model, stats, payload = load_checkpoint(ckpt)
    splits = args.splits or ["validation", "known_test", "unseen_perturbation_test", "harder_test"]
    task = cfg.train.task
    want_behavior = cfg.eval.behavior and hasattr(model, "forward_behavior")
    want_flow = task in ("flow", "joint") and not args.behavior_only
    results = {}
    behavior_results = {}
    for split in splits:
        if split not in datasets:
            raise CliError(f"unknown split {split!r}")
        if want_flow:
            results[split] = evaluate_dataset(
                model,
                datasets[split],
                stats,
                horizons=tuple(cfg.eval.horizons),
                bootstrap_samples=cfg.eval.bootstrap_samples,
                support_swap=cfg.eval.support_swap,
                seed=cfg.run.seed,
            )
        if want_behavior:
            behavior_results[split] = evaluate_behavior(
                model,
                datasets[split],
                stats,
                bootstrap_samples=cfg.eval.bootstrap_samples,
                seed=cfg.run.seed,
                support_swap=cfg.eval.support_swap,
                microbatch=max(cfg.train.microbatch_originals, 1),
            )
    summary = {
        "status": "ok",
        "run_id": cfg.run.run_id,
        "model": cfg.model.name,
        "task": task,
        "seed": cfg.run.seed,
        "checkpoint": str(ckpt),
        "checkpoint_info": checkpoint_info(payload),
        "checkpoint_epoch": payload["epoch"],
        "best_epoch": payload["best_epoch"],
        "params": model.param_report().as_dict(),
        "hashes": payload["hashes"],
        "splits": results,
        "behavior": behavior_results,
        "caveats": [
            "prediction residual을 non-robust probability로 쓰지 않습니다.",
            "큰 prediction error가 곧 non-robust를 뜻하지 않습니다.",
            "4/4 성공은 population robustness 인증이 아닙니다.",
            "unseen perturbation과 harder 일반화는 각각 따로 봅니다.",
            "panel 재정렬 불변성은 pair 정보 사용 여부의 ablation이 아닙니다.",
            "특정 label을 잘 맞혔다고 수학적 구조나 인과성을 발견한 것은 아닙니다.",
        ],
    }
    out = Path(args.out) if args.out else cfg.run_dir / "eval_summary.json"
    atomic_write_json(out, summary)
    return _emit({**summary, "written": str(out)})


def _compare_runs(run_dirs: list[str], out: str | None) -> int:
    """여러 run의 eval_summary.json을 같은 조건 비교표로 모읍니다."""
    rows = []
    for directory in run_dirs:
        path = Path(directory) / "eval_summary.json"
        if not path.exists():
            raise CliError(f"missing eval_summary.json in {directory}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "run_id": payload["run_id"],
            "model": payload["model"],
            "seed": payload.get("seed"),
            "params": payload["params"],
        }
        row["task"] = payload.get("task")
        for split, result in payload.get("splits", {}).items():
            for metric, summary in result["metrics"].items():
                row[f"flow/{split}/{metric}"] = summary["mean"]
        for split, result in payload.get("behavior", {}).items():
            for metric, summary in result["metrics"].items():
                row[f"behavior/{split}/{metric}"] = summary["mean"]
            classification = result.get("robust_classification", {})
            for key in ("accuracy", "balanced_accuracy", "auroc", "brier"):
                if key in classification:
                    row[f"behavior/{split}/{key}"] = classification[key]
        rows.append(row)
    comparison = {
        "status": "ok",
        "rows": rows,
        "note": (
            "같은 input/output/data/loss 조건에서 비교합니다. persistence는 parameter가 없고,"
            " M0는 variant prefix를 받지 않는 점이 구조적 차이입니다."
        ),
    }
    if out:
        atomic_write_json(out, comparison)
        comparison["written"] = out
    return _emit(comparison)


# --------------------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = _load(args)
    datasets = _datasets(cfg)
    ckpt = Path(args.checkpoint) if args.checkpoint else cfg.run_dir / "best.pt"
    if not ckpt.exists():
        raise CliError(f"checkpoint not found: {ckpt}")
    model, stats, _ = load_checkpoint(ckpt)
    dataset = datasets[args.split]
    generator = torch.Generator().manual_seed(cfg.run.seed)
    samples = sample_pairs(dataset, generator, 1, group_indices=[0])
    if args.cut is not None:
        for sample in samples:
            sample.cut = args.cut
    batch = collate(group_by_cut(samples)[0])
    with torch.no_grad():
        pred_norm = model(batch.input_a, stats)
    pred_raw = stats.denorm_target(pred_norm, batch.cut)
    payload = {
        "status": "ok",
        "split": args.split,
        "cut": batch.cut,
        "original_id": batch.original_ids[0],
        "variant_id": batch.variant_ids_a[0],
        "v_hat_shape": list(pred_raw.shape[1:]),
        "v_hat_norm_l2": float(pred_norm[0].norm()),
        "v_hat_raw_l2": float(pred_raw[0].norm()),
        "target_raw_l2": float(batch.target_a[0].norm()),
        "note": "V_hat[d]는 [P, 2, H] native space 값입니다.",
    }
    if args.out:
        torch.save({"v_hat_raw": pred_raw, "cut": batch.cut}, args.out)
        payload["written"] = args.out
    return _emit(payload)


# --------------------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------------------


def cmd_preflight(args: argparse.Namespace) -> int:
    from .adapters.deepmath import probe_deepmath
    from .adapters.mathgap import probe_mathgap
    from .adapters.qwen import probe_qwen, screening_ready, thinking_ready

    cfg = _load(args)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    guard_config(run_dir, cfg.to_dict(), cfg.hash())
    budget = GpuBudget.open(
        run_dir,
        block_minutes=cfg.server.gpu_block_minutes,
        stop_minutes=cfg.server.gpu_stop_minutes,
    )
    can_start, budget_reason = budget.can_start()
    # 선택한 task/데이터 경로에 **필요한 항목만** blocker로 봅니다.
    task = cfg.train.task
    source = cfg.data.source
    pending = []
    mathgap = probe_mathgap()
    if source == "mathgap":
        if not cfg.server.mathgap.generator_path:
            pending.append("mathgap.generator_path is not configured")
        if not mathgap["available"]:
            pending.append("mathgap package is not installed")
    qwen = probe_qwen()
    deepmath = probe_deepmath()
    if source == "deepmath":
        if not cfg.server.deepmath.revision:
            pending.append("server.deepmath.revision (pinned snapshot) is not set")
        if not cfg.server.deepmath.local_path and not deepmath["available"]:
            pending.append("DeepMath snapshot is not available locally")
    if source in ("pages", "deepmath") and not cfg.data.label_path:
        pending.append("data.label_path is not set (run 'aimo build-labels')")
    needs_generation = source in ("deepmath", "pages")
    if needs_generation and not qwen.get("qwen3_ready"):
        pending.append("transformers>=4.51 for Qwen3 is not available")
    thinking = thinking_ready(cfg.server.thinking)
    if needs_generation:
        for name in thinking["needs_calibration"]:
            pending.append(f"thinking profile needs calibration: {name}")
        pending.append("real Qwen3-4B behavior measurement and numerical audit")
    # GPU 실행 가능 여부는 calibration만이 아니라 task의 실제 필수 조건을 함께 봅니다.
    gpu_blockers = [
        item
        for item in pending
        if "calibration" in item or "transformers" in item or "snapshot" in item
    ]
    payload = {
        "status": "ok",
        "dry_run": True,
        "loaded_model_weights": False,
        "run_id": cfg.run.run_id,
        "run_dir": str(run_dir),
        "config_hash": cfg.hash(),
        "device": cfg.run.device,
        "server_paths": {
            "repo_root": cfg.server.repo_root,
            "input_root": cfg.server.input_root,
            "output_root": f"{cfg.server.output_root}/{cfg.run.run_id}",
        },
        "gpu_budget": {
            "can_start": can_start,
            "reason": budget_reason,
            "block_minutes": cfg.server.gpu_block_minutes,
            "stop_minutes": cfg.server.gpu_stop_minutes,
        },
        "thinking_profile": thinking,
        "gpu_full_run_allowed": not gpu_blockers,
        "gpu_blockers": gpu_blockers,
        "checked_for": {"task": task, "data_source": source},
        "legacy_screening_plan": {
            "thinking": cfg.server.screening.thinking,
            "temperature": cfg.server.screening.temperature,
            "top_p": cfg.server.screening.top_p,
            "top_k": cfg.server.screening.top_k,
            "min_p": cfg.server.screening.min_p,
            "max_new_tokens": cfg.server.screening.max_new_tokens,
            "slots_per_prompt": cfg.server.screening.slots_per_prompt,
            "note": "legacy non-thinking 설정입니다. 새 DeepMath 경로에서 쓰지 않습니다.",
        },
        "adapters": {
            "mathgap": mathgap,
            "qwen": qwen,
            "deepmath": deepmath,
            "screening": screening_ready(),
        },
        "task": cfg.train.task,
        "select_metric": cfg.train.resolved_select_metric(),
        "server_pending": pending,
    }
    return _emit(payload)


# --------------------------------------------------------------------------------------
# 서버 단계: prepare-data / screen / extract / run-stage1
# --------------------------------------------------------------------------------------


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """데이터 후보와 split을 준비합니다.

    data.source=deepmath 는 DeepMath primary 경로, pages/synthetic 은 legacy MathGAP
    경로입니다. 실제 대규모 다운로드는 서버에서만 합니다.
    """
    from .adapters import AdapterUnavailable

    cfg = _load(args)
    if cfg.data.source == "deepmath":
        from .adapters.deepmath import (
            DATASET_ID,
            assign_splits,
            load_local_rows,
            probe_deepmath,
            select_candidates,
        )

        prepared = args.prepared or cfg.server.deepmath.prepared_dir
        if prepared:
            # 이미 준비된 registry는 직접 읽고, frozen split을 새로 만들지 않습니다.
            from .adapters.deepmath import load_prepared_registry

            try:
                bundle = load_prepared_registry(
                    prepared, revision=cfg.server.deepmath.revision or "prepared"
                )
            except AdapterUnavailable as exc:
                raise CliError(str(exc)) from exc
            rows = bundle["rows"]
            report = select_candidates(
                rows,
                max_originals=cfg.server.deepmath.max_candidate_originals,
                allowed_topics=tuple(cfg.server.deepmath.allowed_topics),
                require_topic=cfg.server.deepmath.require_topic,
            )
            splits = assign_splits(
                report.candidates, seed=cfg.run.resolved_seeds()["split"],
                frozen=bundle["frozen_splits"],
            )
            out = Path(args.out) if args.out else cfg.run_dir / "deepmath_candidates.json"
            atomic_write_json(
                out,
                {
                    "dataset_id": DATASET_ID,
                    "revision": cfg.server.deepmath.revision,
                    "prepared_dir": bundle["root"],
                    "used_frozen_splits": bundle["frozen_splits"] is not None,
                    "pairs_path": bundle["pairs_path"],
                    "candidates": [row.as_metadata() for row in report.candidates],
                    "splits": {
                        name: [row.row_id for row in rows_] for name, rows_ in splits.items()
                    },
                },
            )
            return _emit(
                {
                    "status": "ok",
                    "source": "deepmath_prepared",
                    "written": str(out),
                    "prepared_dir": bundle["root"],
                    "used_frozen_splits": bundle["frozen_splits"] is not None,
                    "pairs_path": bundle["pairs_path"],
                    "selection": report.as_dict(),
                    "split_sizes": {name: len(v) for name, v in splits.items()},
                    "notes": [
                        "준비된 frozen split을 그대로 썼습니다 (새로 배정하지 않았습니다).",
                        "topic/difficulty는 curator metadata이며 predictor input이 아닙니다.",
                        "r1_solution은 predictor 입력·prompt·label 생성에 쓰지 않습니다.",
                    ],
                }
            )
        local = args.input or cfg.server.deepmath.local_path
        if not local:
            probe = probe_deepmath()
            raise CliError(
                f"{DATASET_ID} snapshot이 필요합니다. --input으로 local JSONL/parquet를 주거나 "
                "server.deepmath.local_path를 설정하세요. 전체 dataset 다운로드는 서버에서만 "
                f"합니다 (SERVER_PENDING, probe={probe['status']})"
            )
        try:
            rows = load_local_rows(
                local,
                revision=cfg.server.deepmath.revision or "local",
                batch_size=cfg.server.deepmath.parquet_batch_size,
            )
        except AdapterUnavailable as exc:
            raise CliError(str(exc)) from exc
        report = select_candidates(
            rows,
            max_originals=cfg.server.deepmath.max_candidate_originals,
            allowed_topics=tuple(cfg.server.deepmath.allowed_topics),
            require_topic=cfg.server.deepmath.require_topic,
        )
        splits = assign_splits(report.candidates, seed=cfg.run.resolved_seeds()["split"])
        out = Path(args.out) if args.out else cfg.run_dir / "deepmath_candidates.json"
        atomic_write_json(
            out,
            {
                "dataset_id": DATASET_ID,
                "revision": cfg.server.deepmath.revision,
                "candidates": [row.as_metadata() for row in report.candidates],
                "splits": {name: [row.row_id for row in rows] for name, rows in splits.items()},
            },
        )
        return _emit(
            {
                "status": "ok",
                "source": "deepmath",
                "written": str(out),
                "selection": report.as_dict(),
                "split_sizes": {name: len(rows) for name, rows in splits.items()},
                "notes": [
                    "후보 수이지 확보된 labeled pair 수가 아닙니다.",
                    "topic/difficulty는 curator metadata이며 predictor input이 아닙니다.",
                    "r1_solution은 predictor 입력·prompt·label 생성에 쓰지 않습니다.",
                    "DeepMath difficulty를 MATH Level 1~5와 같은 척도로 보지 않습니다.",
                    "같은 original과 그 variants/seeds는 한 split에만 둡니다.",
                ],
            }
        )

    from .adapters.mathgap import MathGapAdapter

    try:
        adapter = MathGapAdapter.from_config(cfg.server.mathgap)
    except AdapterUnavailable as exc:
        raise CliError(str(exc)) from exc
    return _emit(
        {
            "status": "ok",
            "source": "mathgap_legacy",
            "revision": adapter.revision,
            "note": "MathGAP/GSM은 구현·저난도 대조용 legacy 경로입니다.",
        }
    )


def cmd_screen(args: argparse.Namespace) -> int:
    from .adapters.qwen import screening_ready

    cfg = _load(args)
    if not args.execute_gpu:
        raise CliError(
            f"screen({cfg.run.run_id})은 실제 GPU 실행이 필요합니다. "
            "계획만 보려면 preflight를 쓰고, 실행하려면 --execute-gpu를 명시하세요 "
            "(SERVER_PENDING)"
        )
    _budget_gate(cfg)
    if not torch.cuda.is_available():
        raise CliError(
            "no CUDA device available; real screening stays SERVER_PENDING on this host. "
            f"{screening_ready()['reason']}"
        )
    raise CliError(
        "SERVER_PENDING: real Qwen3-4B screening is not implemented in this revision; "
        "run it on the server once weights and the MathGAP oracle path are confirmed"
    )


def cmd_extract(args: argparse.Namespace) -> int:
    """Page를 추출합니다. tiny model과 실제 weights가 같은 runner를 씁니다."""
    from .adapters import AdapterUnavailable
    from .adapters.qwen import (
        ExtractionRequest,
        build_tiny_qwen,
        load_real_qwen,
        run_page_extraction,
        select_landmarks,
    )
    from .page import save_pages

    cfg = _load(args)
    budget = GpuBudget.open(
        cfg.run_dir,
        block_minutes=cfg.server.gpu_block_minutes,
        stop_minutes=cfg.server.gpu_stop_minutes,
    )
    guard = None
    if args.tiny:
        try:
            tiny = build_tiny_qwen(hidden_size=args.hidden_size, n_layers=args.layers)
        except AdapterUnavailable as exc:
            raise CliError(str(exc)) from exc
        model, family, real_weights = tiny.model, tiny.family, False
    else:
        if not args.execute_gpu:
            raise CliError(
                "실제 weights 추출은 --execute-gpu가 필요합니다. CPU 검증은 --tiny를 쓰세요"
            )
        _budget_gate(cfg)
        guard = BudgetGuard(budget)
        try:
            model, _tokenizer = load_real_qwen(cfg)
        except AdapterUnavailable as exc:
            raise CliError(str(exc)) from exc
        family, real_weights = cfg.server.model_id, True

    torch.manual_seed(cfg.run.resolved_seeds()["data"])
    prompt_len = args.prompt_len
    requests = []
    for index in range(max(args.n_pages, 1)):
        input_ids = torch.randint(0, 64, (1, prompt_len))
        offsets, valid, rel = select_landmarks(list(range(1, prompt_len, 2)), prompt_len)
        requests.append(
            ExtractionRequest(
                original_id="tiny-orig",
                variant_id=f"tiny-orig#var{prompt_len:03d}_{index:02d}",
                input_ids=input_ids,
                landmark_offsets=offsets,
                valid=valid,
                relative_positions=rel,
            )
        )
    ledger = DedupLedger.open(cfg.run_dir, name="extract_ledger.jsonl")
    try:
        if guard is not None:
            guard.start()
        pages, report = run_page_extraction(
            model,
            requests,
            ledger=ledger,
            guard=guard,
            provenance={
                "source": "tiny_random_init" if args.tiny else "qwen",
                "policy_hash": cfg.server.thinking.protocol_hash(),
                "model_hash": family,
                "tokenizer_hash": "tiny-none" if args.tiny else "SERVER_PENDING",
            },
        )
    finally:
        if guard is not None:
            guard.commit()
    out = Path(args.out) if args.out else cfg.run_dir / "tiny_page.npz"
    if pages:
        save_pages(pages, out)
    payload = {
        "status": "ok" if not report["stopped"] else "stopped",
        "mode": "tiny_random_init" if args.tiny else "real_weights",
        "family": family,
        "real_weights": real_weights,
        "extraction": report,
        "n_pages": len(pages),
        "state_shape": list(pages[0].state.shape) if pages else None,
        "updates_shape": list(pages[0].updates.shape) if pages else None,
        "n_valid_landmarks": int(pages[0].valid.sum()) if pages else None,
        "residual_identity_max_error": (
            max(page.residual_identity_error() for page in pages) if pages else None
        ),
        "n_extracted_total": len(ledger),
        "written": str(out) if pages else None,
        "skipped": report["skipped"],
        "note": (
            "random-init tiny config 검증입니다. 실제 Qwen3-4B 추출은 SERVER_PENDING입니다."
            if args.tiny
            else "실제 weights 경로입니다."
        ),
    }
    return _emit(payload)


def cmd_import_pairs(args: argparse.Namespace) -> int:
    """검증된 original-variant pair를 읽어 freeze합니다."""
    from .adapters import AdapterUnavailable
    from .adapters.perturbation import FrozenPairStore, import_verified_pairs

    cfg = _load(args)
    source = args.input or cfg.data.pair_path
    if not source:
        raise CliError("import-pairs needs --input or data.pair_path")
    try:
        candidates, report = import_verified_pairs(source)
    except AdapterUnavailable as exc:
        raise CliError(str(exc)) from exc
    store = FrozenPairStore.freeze(candidates)
    out = Path(args.out) if args.out else cfg.run_dir / "frozen_pairs.json"
    try:
        store.save(out, overwrite=args.overwrite)
    except AdapterUnavailable as exc:
        raise CliError(str(exc)) from exc
    return _emit(
        {
            "status": "ok",
            "written": str(out),
            "import_report": report,
            "frozen": store.report(),
            "note": "행동 실행 전에 freeze했습니다. 이후 후보를 바꾸면 hash가 달라집니다.",
        }
    )


def _load_plans(path: str) -> list:
    """collection plan JSONL을 읽습니다."""
    from .collect import CollectionPlan

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [CollectionPlan(**row) for row in rows]


def _load_mock_responses(path: str) -> dict:
    """로컬 검증용 mock generation 응답을 읽습니다."""
    from .adapters.qwen import SlotResult

    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["slot_id"]] = SlotResult(**row)
    return out


def cmd_collect_outcomes(args: argparse.Namespace) -> int:
    """행동 측정 결과를 적재하거나, 주입된 backend로 실제 실행 경로를 돕니다.

    로컬에서는 mock backend로만 실행합니다. 실제 weights 로드와 generation은 서버 단계입니다.
    """
    from .adapters import AdapterUnavailable
    from .collect import collect_outcomes

    cfg = _load(args)
    out = Path(args.out) if args.out else cfg.run_dir / "outcomes.json"
    store = OutcomeStore.load(out) if out.exists() else OutcomeStore()

    if args.plans:
        # ---- 실행 경로 (backend 주입) ----
        plans = _load_plans(args.plans)
        profile = cfg.server.thinking
        if args.backend == "qwen":
            if not args.execute_gpu:
                raise CliError(
                    "backend=qwen은 실제 GPU 실행이므로 --execute-gpu가 필요합니다"
                )
            _budget_gate(cfg)
            from .adapters.qwen import QwenGenerationBackend

            try:
                backend = QwenGenerationBackend(profile)
            except AdapterUnavailable as exc:
                raise CliError(str(exc)) from exc
        else:
            if not args.mock_responses:
                raise CliError("backend=mock은 --mock-responses JSONL이 필요합니다")
            from .adapters.qwen import MockGenerationBackend

            backend = MockGenerationBackend(_load_mock_responses(args.mock_responses))
        budget = GpuBudget.open(
            cfg.run_dir,
            block_minutes=cfg.server.gpu_block_minutes,
            stop_minutes=cfg.server.gpu_stop_minutes,
        )
        guard = BudgetGuard(budget) if args.backend == "qwen" else None
        ledger = DedupLedger.open(cfg.run_dir, name="slot_ledger.jsonl")
        try:
            if guard is not None:
                guard.start()
            report = collect_outcomes(
                backend,
                plans,
                profile,
                ledger=ledger,
                guard=guard,
                policy_hash=profile.protocol_hash(),
            )
        except ValueError as exc:
            raise CliError(str(exc)) from exc
        finally:
            if guard is not None:
                guard.commit()
        merge_report = store.merge(report.outcomes, mode=args.merge_mode)
        store.save(out)
        return _emit(
            {
                "status": "ok" if not report.stopped else "stopped",
                "backend": args.backend,
                "written": str(out),
                "collection": report.as_dict(),
                "merge": merge_report,
                "store": store.report(),
                "note": (
                    "mock backend는 로컬 경로 검증 전용입니다. 실제 generation은 "
                    "SERVER_PENDING입니다."
                ),
            }
        )

    if not args.from_file:
        if not args.execute_gpu:
            raise CliError(
                "collect-outcomes는 --from-file로 서버 결과를 적재하거나, --plans와 "
                "backend로 실행 경로를 돌려야 합니다"
            )
        _budget_gate(cfg)
        pending = cfg.server.thinking.needs_calibration()
        if pending:
            raise CliError(
                f"thinking profile is not calibrated yet: {pending}; "
                "fix them in the server calibration manifest before a full GPU run "
                "(SERVER_PENDING)"
            )
        raise CliError("--plans를 주어 실행 계획을 지정하세요 (SERVER_PENDING)")
    incoming = OutcomeStore.from_jsonl(
        args.from_file, fill_missing_as_not_started=args.fill_missing_as_not_started
    )
    try:
        merge_report = store.merge(incoming, mode=args.merge_mode, cohort=args.cohort)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    store.save(out)
    return _emit(
        {
            "status": "ok",
            "written": str(out),
            "merge": merge_report,
            "store": store.report(),
            "note": (
                "fill_not_started는 미시작 slot만 채우고, separate_cohort는 독립 재실행을 "
                "별도로 보존합니다. 기존 X를 새 성공으로 대체하지 않습니다."
            ),
        }
    )


def cmd_build_labels(args: argparse.Namespace) -> int:
    """outcome counts와 freeze된 pair에서 label store를 만듭니다."""
    from .adapters.perturbation import FrozenPairStore

    cfg = _load(args)
    outcome_path = Path(args.outcomes or cfg.data.outcome_path or cfg.run_dir / "outcomes.json")
    pair_path = Path(args.pairs or cfg.data.pair_path or cfg.run_dir / "frozen_pairs.json")
    for path, name in ((outcome_path, "outcomes"), (pair_path, "frozen pairs")):
        if not path.exists():
            raise CliError(f"{name} store not found: {path}")
    outcomes = OutcomeStore.load(outcome_path)
    pairs = FrozenPairStore.load(pair_path)
    policy = BinaryRobustPolicy(
        enabled=cfg.data.robust_policy.enabled,
        definition_id=cfg.data.robust_policy.definition_id,
        source=cfg.data.robust_policy.source,
    )
    try:
        store = build_label_store(outcomes, pairs.candidates, robust_policy=policy)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    out = Path(args.out) if args.out else cfg.run_dir / "labels.json"
    store.save(out)
    report = store.coverage_report()
    return _emit(
        {
            "status": "ok",
            "written": str(out),
            "coverage": report,
            "robust_policy": {
                "enabled": policy.enabled,
                "definition_id": policy.definition_id,
                "source": policy.source,
            },
            "notes": [
                "C4 필터를 적용하지 않았습니다. 성공/실패/유지/개선 사례를 모두 보존합니다.",
                "미확정 pair는 behavior loss에서 제외되지만 flow에는 쓸 수 있습니다.",
                "robust label은 frozen definition이 있을 때만 채워집니다.",
            ],
        }
    )


def cmd_predict_behavior(args: argparse.Namespace) -> int:
    """checkpoint로 pair drop과 panel robust probability를 예측합니다."""
    cfg = _load(args)
    datasets = _datasets(cfg)
    ckpt = Path(args.checkpoint) if args.checkpoint else cfg.run_dir / "best.pt"
    if not ckpt.exists():
        raise CliError(f"checkpoint not found: {ckpt}")
    model, stats, payload = load_checkpoint(ckpt)
    if not hasattr(model, "forward_behavior"):
        raise CliError(
            f"checkpoint model {payload.get('model_name')!r} has no behavior head; "
            "train a behavior or joint task first"
        )
    dataset = datasets[args.split]
    groups = dataset.groups[: max(args.n_originals, 1)]
    batch = collate_panels(groups)
    with torch.no_grad():
        out = model.forward_behavior(batch.inputs, stats)
        panel = model.panel_outputs(out, batch.pair_panel, batch.pair_slot, batch.panel_mask)
    trained = model.trained_heads() if hasattr(model, "trained_heads") else {}
    robust_trained = bool(trained.get("robust", False))
    drop_trained = bool(trained.get("pair_drop", False))
    panels = []
    debug_panels = []
    for b, group in enumerate(groups):
        members = []
        for slot, variant_id in enumerate(batch.variant_ids[b]):
            raw_drop = float(panel.pair_drop[b, slot])
            members.append(
                {
                    "variant_id": variant_id,
                    # 학습되지 않은 head의 원시 score는 canonical 출력에 넣지 않습니다.
                    "pair_drop_hat": raw_drop if drop_trained else None,
                    "pair_drop_label": (
                        float(batch.drop_target[b, slot])
                        if bool(batch.drop_mask[b, slot])
                        else None
                    ),
                    "pooling_weight": float(panel.pooling_weights[b, slot]),
                }
            )
        panels.append(
            {
                "original_id": group.original_id,
                "panel_id": group.panel_id,
                # robust head가 학습되지 않았으면 null/untrained로 표시합니다.
                "robust_probability": float(panel.robust_prob[b]) if robust_trained else None,
                "robust_head_status": "trained" if robust_trained else "untrained",
                "robust_label": (
                    int(batch.robust_target[b]) if bool(batch.robust_mask[b]) else None
                ),
                # pair 예측에서 계산한 diagnostic이며 별도 학습 head가 아닙니다.
                "panel_max_drop_diagnostic": (
                    float(panel.max_drop[b]) if drop_trained else None
                ),
                "members": members,
            }
        )
        debug_panels.append(
            {
                "original_id": group.original_id,
                "raw_robust_probability": float(panel.robust_prob[b]),
                "raw_pair_drop": [
                    float(panel.pair_drop[b, slot])
                    for slot in range(len(batch.variant_ids[b]))
                ],
            }
        )
    payload_out = {
        "status": "ok",
        "split": args.split,
        "checkpoint": str(ckpt),
        "checkpoint_info": checkpoint_info(payload),
        "trained_heads": trained,
        "untrained_heads": [name for name, ok in trained.items() if not ok],
        "panels": panels,
        "max_drop_source": "derived_from_pair_predictions",
        "caveats": [
            "prediction residual은 robustness 지표가 아닙니다.",
            "학습되지 않은 head의 출력은 검증된 robust probability가 아닙니다.",
            "pair score와 pooling weight는 개별 변형의 causal importance가 아닙니다.",
        ],
    }
    if not robust_trained or not drop_trained:
        payload_out["warning"] = (
            "one or more heads were never trained on a real label; canonical outputs are null "
            "and raw scores are only in debug_raw_scores"
        )
        payload_out["debug_raw_scores"] = debug_panels
    if args.out:
        atomic_write_json(args.out, payload_out)
        payload_out["written"] = args.out
    return _emit(payload_out)


def cmd_resume(args: argparse.Namespace) -> int:
    """같은 config로 train을 이어서 실행합니다."""
    args.resume = True
    return cmd_train(args)


def cmd_run_stage1(args: argparse.Namespace) -> int:
    cfg = _load(args)
    stages = args.stages or ["prepare-data", "screen", "extract", "train"]
    executed, failed = [], None
    with run_lock(cfg.run_dir):
        for stage in stages:
            try:
                if stage == "prepare-data":
                    cmd_prepare_data(args)
                elif stage == "screen":
                    cmd_screen(args)
                elif stage == "extract":
                    cmd_extract(args)
                elif stage == "train":
                    datasets = _datasets(cfg)
                    train(cfg, datasets, resume=args.resume)
                else:
                    raise CliError(f"unknown stage {stage!r}")
            except CliError as exc:
                failed = {"stage": stage, "reason": str(exc)}
                break  # 미지원 단계에서 즉시 멈추고 빈 성공 파일을 남기지 않습니다.
            executed.append(stage)
    payload = {
        "status": "ok" if failed is None else "stopped",
        "executed": executed,
        "stopped_at": failed,
    }
    if failed is not None:
        _emit(payload)
        return 2
    return _emit(payload)


# --------------------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aimo", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="store_true", help="print the aimo version and exit")
    sub = parser.add_subparsers(dest="command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="dotted config override")
        p.add_argument("--run-id", default=None, help="override run.run_id")
        p.add_argument(
            "--execute-gpu",
            action="store_true",
            help="실제 accelerator 실행을 명시적으로 허용합니다 (기본은 dry-run/CPU)",
        )

    p_check = sub.add_parser("check", help="환경과 데이터 계약 self-check")
    common(p_check)
    p_check.set_defaults(func=cmd_check)

    p_toy = sub.add_parser("make-toy", help="E0 synthetic Page 생성")
    common(p_toy)
    p_toy.add_argument("--out", default=None, help="출력 directory")
    p_toy.set_defaults(func=cmd_make_toy)

    p_train = sub.add_parser("train", help="behavior / flow / joint 학습")
    common(p_train)
    p_train.add_argument("--resume", action="store_true", help="last.pt에서 이어서 학습")
    p_train.set_defaults(func=cmd_train)

    p_resume = sub.add_parser("resume", help="같은 config로 train 이어서 실행")
    common(p_resume)
    p_resume.set_defaults(func=cmd_resume, resume=True)

    p_eval = sub.add_parser("evaluate", help="checkpoint 평가 또는 run 비교")
    common(p_eval)
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--splits", nargs="*", default=None)
    p_eval.add_argument("--compare", nargs="*", default=None, help="비교할 run directory 목록")
    p_eval.add_argument("--behavior-only", action="store_true", help="flow 지표를 생략합니다")
    p_eval.add_argument("--out", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="checkpoint로 V_hat 예측")
    common(p_pred)
    p_pred.add_argument("--checkpoint", default=None)
    p_pred.add_argument("--split", default="known_test")
    p_pred.add_argument("--cut", type=int, default=None)
    p_pred.add_argument("--out", default=None)
    p_pred.set_defaults(func=cmd_predict)

    p_predb = sub.add_parser("predict-behavior", help="pair drop / panel robust probability 예측")
    common(p_predb)
    p_predb.add_argument("--checkpoint", default=None)
    p_predb.add_argument("--split", default="known_test")
    p_predb.add_argument("--n-originals", type=int, default=3)
    p_predb.add_argument("--out", default=None)
    p_predb.set_defaults(func=cmd_predict_behavior)

    p_pre = sub.add_parser("preflight", help="서버 실행 전 dry-run")
    common(p_pre)
    p_pre.set_defaults(func=cmd_preflight)

    p_prep = sub.add_parser("prepare-data", help="DeepMath 후보/split 준비 (legacy: MathGAP)")
    common(p_prep)
    p_prep.add_argument("--input", default=None, help="local DeepMath JSONL/parquet snapshot")
    p_prep.add_argument(
        "--prepared", default=None, help="준비된 registry directory (frozen split 사용)"
    )
    p_prep.add_argument("--out", default=None)
    p_prep.set_defaults(func=cmd_prepare_data)

    p_pairs = sub.add_parser("import-pairs", help="검증된 original-variant pair import 후 freeze")
    common(p_pairs)
    p_pairs.add_argument("--input", default=None, help="검증된 pair JSONL")
    p_pairs.add_argument("--out", default=None)
    p_pairs.add_argument(
        "--overwrite", action="store_true", help="freeze된 후보를 의도적으로 다시 freeze"
    )
    p_pairs.set_defaults(func=cmd_import_pairs)

    p_outcomes = sub.add_parser("collect-outcomes", help="행동 측정 결과(outcome counts) 적재")
    common(p_outcomes)
    p_outcomes.add_argument("--from-file", default=None, help="서버가 만든 outcome JSONL")
    p_outcomes.add_argument(
        "--merge-mode",
        default="new_only",
        choices=list(OutcomeStore.MERGE_MODES),
        help="기존 기록과의 병합 방식",
    )
    p_outcomes.add_argument("--cohort", default=None, help="separate_cohort에 쓸 cohort 이름")
    p_outcomes.add_argument(
        "--fill-missing-as-not-started",
        action="store_true",
        help="명시적 import 규칙: 미기록 planned slot을 not_started로 채웁니다",
    )
    p_outcomes.add_argument("--plans", default=None, help="collection plan JSONL")
    p_outcomes.add_argument(
        "--backend", default="mock", choices=["mock", "qwen"], help="generation backend"
    )
    p_outcomes.add_argument("--mock-responses", default=None, help="mock backend 응답 JSONL")
    p_outcomes.add_argument("--out", default=None)
    p_outcomes.set_defaults(func=cmd_collect_outcomes)

    p_labels = sub.add_parser("build-labels", help="outcome counts -> pair drop / panel label")
    common(p_labels)
    p_labels.add_argument("--outcomes", default=None)
    p_labels.add_argument("--pairs", default=None)
    p_labels.add_argument("--out", default=None)
    p_labels.set_defaults(func=cmd_build_labels)

    p_screen = sub.add_parser("screen", help="legacy screening (--execute-gpu 필요)")
    common(p_screen)
    p_screen.set_defaults(func=cmd_screen)

    p_extract = sub.add_parser("extract", help="Page 추출")
    common(p_extract)
    p_extract.add_argument("--tiny", action="store_true", help="random-init tiny config CPU 검증")
    p_extract.add_argument("--hidden-size", type=int, default=32)
    p_extract.add_argument("--layers", type=int, default=4)
    p_extract.add_argument("--prompt-len", type=int, default=40)
    p_extract.add_argument("--n-pages", type=int, default=1)
    p_extract.add_argument("--out", default=None)
    p_extract.set_defaults(func=cmd_extract)

    p_stage1 = sub.add_parser("run-stage1", help="legacy flow 경로 단계 실행")
    common(p_stage1)
    p_stage1.add_argument("--stages", nargs="*", default=None)
    p_stage1.add_argument("--resume", action="store_true")
    p_stage1.add_argument("--tiny", action="store_true")
    p_stage1.add_argument("--hidden-size", type=int, default=32)
    p_stage1.add_argument("--layers", type=int, default=4)
    p_stage1.add_argument("--prompt-len", type=int, default=40)
    p_stage1.add_argument("--n-pages", type=int, default=1)
    p_stage1.add_argument("--out", default=None)
    p_stage1.set_defaults(func=cmd_run_stage1)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__

        print(__version__)
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (CliError, GpuBudgetExceeded) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
