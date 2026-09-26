"""Training loop: AdamW, gradient accumulation, validation 기반 checkpoint 선택.

기본값은 lr 3e-4, weight_decay 1e-3, effective batch 8 originals, gradient clip 1,
최대 100 epochs, validation patience 15입니다. checkpoint 선택에는 validation만
사용합니다. checkpoint에는 config/data/split/stats hash와 optimizer/RNG state를
함께 저장해 resume이 재현되게 합니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .config import Config
from .data import (
    NormStats,
    PageDataset,
    collate,
    compute_norm_stats,
    enumerate_pairs,
    group_by_cut,
    sample_pairs,
)
from .losses import compute_loss
from .model import build_model
from .runtime import atomic_save, atomic_write_json, guard_config

CHECKPOINT_LAST = "last.pt"
CHECKPOINT_BEST = "best.pt"


@dataclass
class EpochRecord:
    epoch: int
    train_total: float
    val_total: float
    val_next: float
    val_within: float
    val_roll: float
    seconds: float


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _split_hashes(datasets: dict[str, PageDataset]) -> dict[str, str]:
    return {name: dataset.data_hash() for name, dataset in datasets.items()}


@torch.no_grad()
def validation_loss(
    model: nn.Module, dataset: PageDataset, stats: NormStats, cfg: Config
) -> dict[str, float]:
    """결정적 validation loss. 모든 cut과 sibling 조합을 고정 순서로 나열합니다."""
    model.eval()
    cuts = list(range(dataset.n_blocks))
    samples = enumerate_pairs(dataset, cuts)
    totals = {"total": 0.0, "L_next": 0.0, "L_within": 0.0, "L_roll": 0.0}
    weight = 0.0
    for bucket in group_by_cut(samples):
        for chunk in _chunk(bucket, max(cfg.train.microbatch_originals, 1)):
            batch = collate(chunk)
            breakdown = compute_loss(
                model,
                batch,
                stats,
                w_next=cfg.train.w_next,
                w_within=cfg.train.w_within,
                w_roll=cfg.train.w_roll,
                horizons=tuple(cfg.train.horizons),
            )
            values = breakdown.as_dict()
            n = len(chunk)
            for key in totals:
                value = values[key]
                totals[key] += (0.0 if value != value else value) * n
            weight += n
    return {key: value / max(weight, 1.0) for key, value in totals.items()}


def train(
    cfg: Config,
    datasets: dict[str, PageDataset],
    resume: bool = False,
) -> dict:
    """train split으로 학습하고 validation으로 checkpoint를 고릅니다."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    guard_config(run_dir, cfg.to_dict(), cfg.hash())

    torch.manual_seed(cfg.run.seed)
    train_set = datasets["train"].subset(cfg.data.subset_fraction)
    val_set = datasets["validation"]
    # normalization은 각 training subset 안에서만 계산합니다.
    stats = compute_norm_stats(train_set, floor=cfg.data.scale_floor)

    model = build_model(cfg, train_set.hidden_size, train_set.n_blocks, train_set.n_landmarks)
    report = model.param_report()
    meta = {
        "hidden_size": train_set.hidden_size,
        "n_blocks": train_set.n_blocks,
        "n_landmarks": train_set.n_landmarks,
    }
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = (
        torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        if params
        else None
    )

    hashes = {
        "config_hash": cfg.hash(),
        "stats_hash": stats.hash(),
        "split_hashes": _split_hashes(datasets),
        "train_subset_hash": train_set.data_hash(),
        "subset_fraction": cfg.data.subset_fraction,
    }

    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    history: list[EpochRecord] = []
    sampler = torch.Generator().manual_seed(cfg.run.seed + 101)

    last_path = run_dir / CHECKPOINT_LAST
    if resume and last_path.exists():
        payload = torch.load(last_path, map_location="cpu")
        _check_hashes(payload["hashes"], hashes)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng_state"])
        sampler.set_state(payload["sampler_rng_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_val = float(payload["best_val"])
        best_epoch = int(payload["best_epoch"])
        history = [EpochRecord(**record) for record in payload.get("history", [])]

    n_groups = len(train_set.groups)
    batch_size = max(cfg.train.batch_originals, 1)
    micro = max(cfg.train.microbatch_originals, 1)
    patience_left = cfg.train.patience - (len(history) - 1 - best_epoch if history else 0)

    for epoch in range(start_epoch, cfg.train.max_epochs):
        started = time.time()
        model.train()
        order = torch.randperm(n_groups, generator=sampler).tolist()
        epoch_total, epoch_weight = 0.0, 0.0
        for group_chunk in _chunk(order, batch_size):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            samples = sample_pairs(train_set, sampler, cfg.train.cuts_per_pair, group_chunk)
            buckets = [
                chunk for bucket in group_by_cut(samples) for chunk in _chunk(bucket, micro)
            ]
            total_items = sum(len(chunk) for chunk in buckets)
            for chunk in buckets:
                batch = collate(chunk)
                breakdown = compute_loss(
                    model,
                    batch,
                    stats,
                    w_next=cfg.train.w_next,
                    w_within=cfg.train.w_within,
                    w_roll=cfg.train.w_roll,
                    horizons=tuple(cfg.train.horizons),
                )
                scaled = breakdown.total * (len(chunk) / total_items)
                if optimizer is not None and scaled.requires_grad:
                    scaled.backward()
                epoch_total += float(breakdown.total) * len(chunk)
                epoch_weight += len(chunk)
            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                optimizer.step()

        val = validation_loss(model, val_set, stats, cfg)
        record = EpochRecord(
            epoch=epoch,
            train_total=epoch_total / max(epoch_weight, 1.0),
            val_total=val["total"],
            val_next=val["L_next"],
            val_within=val["L_within"],
            val_roll=val["L_roll"],
            seconds=time.time() - started,
        )
        history.append(record)

        improved = val["total"] < best_val - 1e-9
        if improved:
            best_val, best_epoch = val["total"], epoch
            patience_left = cfg.train.patience
            atomic_save(
                _checkpoint_payload(
                    model, optimizer, stats, cfg, hashes, epoch, best_val, best_epoch,
                    history, sampler, meta,
                ),
                run_dir / CHECKPOINT_BEST,
            )
        else:
            patience_left -= 1
        atomic_save(
            _checkpoint_payload(
                model, optimizer, stats, cfg, hashes, epoch, best_val, best_epoch,
                history, sampler, meta,
            ),
            last_path,
        )
        if optimizer is None:
            break  # parameter가 없는 baseline은 한 번의 평가로 끝냅니다.
        if patience_left <= 0:
            break

    summary = {
        "run_id": cfg.run.run_id,
        "model": cfg.model.name,
        "seed": cfg.run.seed,
        "params": report.as_dict(),
        "hashes": hashes,
        "best_epoch": best_epoch,
        "best_val_total": None if best_val == float("inf") else best_val,
        "epochs_run": len(history),
        "history": [record.__dict__ for record in history],
        "n_train_originals": len(train_set.groups),
        "n_validation_originals": len(val_set.groups),
    }
    atomic_write_json(run_dir / "train_summary.json", summary)
    return summary


def _checkpoint_payload(
    model: nn.Module,
    optimizer,
    stats: NormStats,
    cfg: Config,
    hashes: dict,
    epoch: int,
    best_val: float,
    best_epoch: int,
    history: list[EpochRecord],
    sampler: torch.Generator,
    meta: dict,
) -> dict:
    return {
        "model": model.state_dict(),
        "model_meta": meta,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "norm_stats": stats.state_dict(),
        "config": cfg.to_dict(),
        "hashes": hashes,
        "epoch": epoch,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "history": [record.__dict__ for record in history],
        "torch_rng_state": torch.get_rng_state(),
        "sampler_rng_state": sampler.get_state(),
        "format_version": 1,
    }


def _check_hashes(stored: dict, current: dict) -> None:
    for key in ("config_hash", "stats_hash", "train_subset_hash"):
        if stored.get(key) != current.get(key):
            raise ValueError(
                f"checkpoint {key} mismatch: stored {stored.get(key)} vs current {current.get(key)}"
            )
    if stored.get("split_hashes") != current.get("split_hashes"):
        raise ValueError("checkpoint split hashes do not match the loaded dataset")


def load_checkpoint(
    path: str | Path, cfg: Config | None = None
) -> tuple[nn.Module, NormStats, dict]:
    """checkpoint를 다시 읽어 model과 normalization stats를 복원합니다.

    model shape는 checkpoint에 저장한 model_meta에서 그대로 가져옵니다.
    """
    from .config import config_from_dict  # 지역 import로 순환 참조를 피합니다.

    payload = torch.load(Path(path), map_location="cpu")
    stored_cfg = config_from_dict(payload["config"])
    if cfg is not None and cfg.hash() != payload["hashes"]["config_hash"]:
        raise ValueError(
            "config hash mismatch between checkpoint and requested config; "
            "load the checkpoint with its own config"
        )
    meta = payload["model_meta"]
    stats = NormStats.from_state_dict(payload["norm_stats"])
    model = build_model(
        stored_cfg,
        hidden_size=int(meta["hidden_size"]),
        n_blocks=int(meta["n_blocks"]),
        n_landmarks=int(meta["n_landmarks"]),
    )
    model.load_state_dict(payload["model"])
    model.eval()
    return model, stats, payload
