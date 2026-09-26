"""aimo CLI. python -m aimo 도 같은 진입점을 씁니다.

명령
  check         환경과 데이터 계약 self-check
  make-toy      E0 synthetic Page 생성
  train         Stage 1 predictor 학습
  evaluate      checkpoint 평가, 또는 여러 run 비교
  predict       checkpoint로 한 batch V_hat 예측
  preflight     서버 실행 전 dry-run (model weights를 읽지 않습니다)
  prepare-data  MathGAP 자료 준비 (adapter 설정 필요)
  screen        실제 screening (--execute-gpu 필요)
  extract       Page 추출 (--tiny는 CPU random-init 검증, 실제 weights는 --execute-gpu)
  run-stage1    prepare-data -> screen -> extract -> train 순서 실행

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
    collate,
    compute_norm_stats,
    group_by_cut,
    load_dataset,
    make_synthetic_dataset,
    sample_pairs,
    save_dataset,
)
from .evaluate import evaluate_dataset
from .model import build_model
from .runtime import (
    DedupLedger,
    GpuBudget,
    GpuBudgetExceeded,
    atomic_write_json,
    guard_config,
    run_lock,
)
from .train import load_checkpoint, train

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
    if cfg.data.source == "pages":
        if not cfg.data.page_dir:
            raise CliError("data.source=pages requires data.page_dir")
        page_dir = Path(cfg.data.page_dir)
        if not (page_dir / "index.json").exists():
            raise CliError(
                f"no extracted pages at {page_dir} (missing index.json); "
                "run 'aimo extract' first, or point data.page_dir at an existing page directory"
            )
        return load_dataset(page_dir)
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
    from .adapters.mathgap import probe_mathgap
    from .adapters.qwen import probe_qwen
    from .page import Page

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
        "adapters": {"mathgap": probe_mathgap(), "qwen": probe_qwen()},
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
    results = {}
    for split in splits:
        if split not in datasets:
            raise CliError(f"unknown split {split!r}")
        results[split] = evaluate_dataset(
            model,
            datasets[split],
            stats,
            horizons=tuple(cfg.eval.horizons),
            bootstrap_samples=cfg.eval.bootstrap_samples,
            support_swap=cfg.eval.support_swap,
            seed=cfg.run.seed,
        )
    summary = {
        "status": "ok",
        "run_id": cfg.run.run_id,
        "model": cfg.model.name,
        "seed": cfg.run.seed,
        "checkpoint": str(ckpt),
        "checkpoint_epoch": payload["epoch"],
        "best_epoch": payload["best_epoch"],
        "params": model.param_report().as_dict(),
        "hashes": payload["hashes"],
        "splits": results,
        "caveats": [
            "큰 prediction error가 곧 non-robust를 뜻하지 않습니다.",
            "4/4 성공은 population robustness 인증이 아닙니다.",
            "unseen perturbation과 harder 일반화는 각각 따로 봅니다.",
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
        for split, result in payload["splits"].items():
            for metric, summary in result["metrics"].items():
                row[f"{split}/{metric}"] = summary["mean"]
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
    from .adapters.mathgap import probe_mathgap
    from .adapters.qwen import probe_qwen, screening_ready

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
    pending = []
    mathgap = probe_mathgap()
    if not cfg.server.mathgap.generator_path:
        pending.append("mathgap.generator_path is not configured")
    if not mathgap["available"]:
        pending.append("mathgap package is not installed")
    qwen = probe_qwen()
    if not qwen.get("qwen3_ready"):
        pending.append("transformers>=4.51 for Qwen3 is not available")
    pending.append("real Qwen3-4B screening and numerical audit")
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
        "screening_plan": {
            "thinking": cfg.server.screening.thinking,
            "temperature": cfg.server.screening.temperature,
            "top_p": cfg.server.screening.top_p,
            "top_k": cfg.server.screening.top_k,
            "min_p": cfg.server.screening.min_p,
            "max_new_tokens": cfg.server.screening.max_new_tokens,
            "slots_per_prompt": cfg.server.screening.slots_per_prompt,
            "note": "연구용 screening 설정이며 공식 AIMO 평가 정책이 아닙니다.",
        },
        "adapters": {"mathgap": mathgap, "qwen": qwen, "screening": screening_ready()},
        "server_pending": pending,
    }
    return _emit(payload)


# --------------------------------------------------------------------------------------
# 서버 단계: prepare-data / screen / extract / run-stage1
# --------------------------------------------------------------------------------------


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from .adapters import AdapterUnavailable
    from .adapters.mathgap import MathGapAdapter

    cfg = _load(args)
    try:
        adapter = MathGapAdapter.from_config(cfg.server.mathgap)
    except AdapterUnavailable as exc:
        raise CliError(str(exc)) from exc
    payload = {
        "status": "ok",
        "revision": adapter.revision,
        "note": "generator/renderer/oracle 경로가 확인된 경우에만 여기까지 옵니다.",
    }
    return _emit(payload)


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
    from .adapters import AdapterUnavailable
    from .adapters.qwen import build_tiny_qwen, extract_page, select_landmarks

    cfg = _load(args)
    if not args.tiny:
        if not args.execute_gpu:
            raise CliError(
                "실제 weights 추출은 --execute-gpu가 필요합니다. CPU 검증은 --tiny를 쓰세요"
            )
        _budget_gate(cfg)
        raise CliError(
            "SERVER_PENDING: real Qwen3-4B extraction runs on the server; "
            "이 저장소는 로컬에서 4B weights를 내려받지 않습니다"
        )
    try:
        tiny = build_tiny_qwen(hidden_size=args.hidden_size, n_layers=args.layers)
    except AdapterUnavailable as exc:
        raise CliError(str(exc)) from exc
    from .page import save_pages

    # dedup/resume: 이미 추출한 prompt는 다시 돌리지 않습니다.
    ledger = DedupLedger.open(cfg.run_dir, name="extract_ledger.jsonl")
    variant_id = f"tiny-orig#var{args.prompt_len:03d}"
    out = Path(args.out) if args.out else cfg.run_dir / "tiny_page.npz"
    if ledger.seen(variant_id) and out.exists():
        return _emit(
            {
                "status": "ok",
                "mode": "tiny_random_init",
                "skipped": True,
                "reason": "already extracted (dedup ledger)",
                "variant_id": variant_id,
                "written": str(out),
            }
        )
    torch.manual_seed(cfg.run.seed)
    prompt_len = args.prompt_len
    input_ids = torch.randint(0, 64, (1, prompt_len))
    offsets, valid, rel = select_landmarks(list(range(1, prompt_len, 2)), prompt_len)
    page = extract_page(tiny.model, input_ids, offsets, valid, rel, "tiny-orig", variant_id)
    save_pages([page], out)
    ledger.mark(variant_id, {"prompt_len": prompt_len, "family": tiny.family})
    payload = {
        "status": "ok",
        "mode": "tiny_random_init",
        "skipped": False,
        "family": tiny.family,
        "real_weights": False,
        "variant_id": variant_id,
        "state_shape": list(page.state.shape),
        "updates_shape": list(page.updates.shape),
        "n_valid_landmarks": int(page.valid.sum()),
        "residual_identity_max_error": page.residual_identity_error(),
        "n_extracted_total": len(ledger),
        "written": str(out),
        "note": "random-init tiny config 검증입니다. 실제 Qwen3-4B 추출은 SERVER_PENDING입니다.",
    }
    return _emit(payload)


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

    p_train = sub.add_parser("train", help="Stage 1 predictor 학습")
    common(p_train)
    p_train.add_argument("--resume", action="store_true", help="last.pt에서 이어서 학습")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="checkpoint 평가 또는 run 비교")
    common(p_eval)
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--splits", nargs="*", default=None)
    p_eval.add_argument("--compare", nargs="*", default=None, help="비교할 run directory 목록")
    p_eval.add_argument("--out", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="checkpoint로 V_hat 예측")
    common(p_pred)
    p_pred.add_argument("--checkpoint", default=None)
    p_pred.add_argument("--split", default="known_test")
    p_pred.add_argument("--cut", type=int, default=None)
    p_pred.add_argument("--out", default=None)
    p_pred.set_defaults(func=cmd_predict)

    p_pre = sub.add_parser("preflight", help="서버 실행 전 dry-run")
    common(p_pre)
    p_pre.set_defaults(func=cmd_preflight)

    p_prep = sub.add_parser("prepare-data", help="MathGAP 자료 준비")
    common(p_prep)
    p_prep.set_defaults(func=cmd_prepare_data)

    p_screen = sub.add_parser("screen", help="실제 screening (--execute-gpu 필요)")
    common(p_screen)
    p_screen.set_defaults(func=cmd_screen)

    p_extract = sub.add_parser("extract", help="Page 추출")
    common(p_extract)
    p_extract.add_argument("--tiny", action="store_true", help="random-init tiny config CPU 검증")
    p_extract.add_argument("--hidden-size", type=int, default=32)
    p_extract.add_argument("--layers", type=int, default=4)
    p_extract.add_argument("--prompt-len", type=int, default=40)
    p_extract.add_argument("--out", default=None)
    p_extract.set_defaults(func=cmd_extract)

    p_stage1 = sub.add_parser("run-stage1", help="Stage 1 단계 순서 실행")
    common(p_stage1)
    p_stage1.add_argument("--stages", nargs="*", default=None)
    p_stage1.add_argument("--resume", action="store_true")
    p_stage1.add_argument("--tiny", action="store_true")
    p_stage1.add_argument("--hidden-size", type=int, default=32)
    p_stage1.add_argument("--layers", type=int, default=4)
    p_stage1.add_argument("--prompt-len", type=int, default=40)
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
