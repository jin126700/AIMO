"""Training loop: behavior / flow / joint task, gradient accumulation, checkpoint 선택.

기본값은 lr 3e-4, weight_decay 1e-3, effective batch 8 originals, gradient clip 1,
최대 100 epochs, validation patience 15입니다. checkpoint 선택에는 validation만
사용하며 어떤 지표를 쓰는지는 시작 전에 config(`train.select_metric`)로 고정합니다.
checkpoint에는 schema/model/task version, config/data/split/stats hash,
optimizer/RNG state, head 학습 여부를 함께 저장해 resume과 재로딩이 재현되게 합니다.

joint task에서는 behavior forward와 flow forward의 gradient가 **같은 LoopedCore**에
누적됩니다. shared parameter는 optimizer에 한 번만 등록되고, flow gradient를 detach하지
않습니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .config import Config
from .data import (
    DATA_SCHEMA_VERSION,
    NormStats,
    PageDataset,
    collate,
    collate_panels,
    compute_norm_stats,
    enumerate_pairs,
    group_by_cut,
    sample_pairs,
)
from .losses import behavior_loss, compute_loss
from .model import build_model
from .runtime import atomic_save, atomic_write_json, guard_config

CHECKPOINT_LAST = "last.pt"
CHECKPOINT_BEST = "best.pt"

# checkpoint schema 버전. v1(flow-only) checkpoint를 새 모델로 조용히 해석하지 않습니다.
CHECKPOINT_SCHEMA_VERSION = "aimo-checkpoint-v2"


@dataclass
class EpochRecord:
    epoch: int
    train_total: float
    val_total: float
    val_next: float
    val_within: float
    val_roll: float
    seconds: float
    val_behavior_total: float = float("nan")
    val_robust: float = float("nan")
    val_pair_drop: float = float("nan")
    val_max_drop: float = float("nan")
    val_flow_total: float = float("nan")


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _split_hashes(datasets: dict[str, PageDataset]) -> dict[str, str]:
    return {name: dataset.data_hash() for name, dataset in datasets.items()}


def _label_coverage(dataset: PageDataset) -> dict:
    """behavior label coverage와 제외율. 실제로 감독된 양을 그대로 보고합니다."""
    n_pairs = 0
    n_drop = 0
    reasons: dict[str, int] = {}
    for group in dataset.groups:
        for variant in group.variants:
            label = group.pair_labels.get(variant.variant_id)
            n_pairs += 1
            if label is not None and label.has_drop:
                n_drop += 1
            else:
                reason = (
                    label.exclusion_reason if label is not None else "no_label"
                ) or "unknown"
                reasons[reason] = reasons.get(reason, 0) + 1
    panels = [group.panel_label for group in dataset.groups]
    return {
        "n_pairs": n_pairs,
        "n_pairs_with_drop": n_drop,
        "pair_exclusion_rate": (1.0 - n_drop / n_pairs) if n_pairs else 0.0,
        "pair_exclusion_reasons": reasons,
        "n_panels": len(panels),
        "n_panels_with_robust": sum(1 for p in panels if p is not None and p.has_robust),
        "n_panels_with_panel_only_max_drop": sum(
            1 for p in panels if p is not None and p.has_max_drop and p.max_drop_is_panel_only
        ),
    }


def _check_label_policy(datasets: dict[str, PageDataset]) -> str:
    """서로 다른 model/thinking/sampling policy의 label을 섞지 않도록 막습니다."""
    hashes = set()
    for dataset in datasets.values():
        for group in dataset.groups:
            for label in group.pair_labels.values():
                if label.policy_hash:
                    hashes.add(label.policy_hash)
    if len(hashes) > 1:
        raise ValueError(
            "datasets mix behavior labels from different model/thinking/sampling policies: "
            f"{sorted(hashes)}; keep one policy per training run"
        )
    return next(iter(hashes), "")


def _behavior_terms(model, cfg: Config, stats: NormStats, groups: list):
    """behavior loss를 한 microbatch에 대해 계산합니다."""
    batch = collate_panels(groups)
    return behavior_loss(
        model,
        batch,
        stats,
        w_robust=cfg.train.w_robust,
        w_pair_drop=cfg.train.w_pair_drop,
        w_max_drop=cfg.train.w_max_drop,
        use_max_drop=cfg.train.use_max_drop,
    )


def _flow_terms(model, cfg: Config, stats: NormStats, samples: list):
    """flow loss를 한 microbatch(단일 cut)에 대해 계산합니다."""
    batch = collate(samples)
    return compute_loss(
        model,
        batch,
        stats,
        w_next=cfg.train.w_next,
        w_within=cfg.train.w_within,
        w_roll=cfg.train.w_roll,
        horizons=tuple(cfg.train.horizons),
    )


@torch.no_grad()
def validation_metrics(
    model: nn.Module, dataset: PageDataset, stats: NormStats, cfg: Config
) -> dict[str, float]:
    """결정적 validation 지표. task에 따라 behavior/flow 항을 모읍니다.

    behavior는 panel 순서대로, flow는 모든 cut과 sibling 조합을 고정 순서로 나열합니다.
    """
    model.eval()
    micro = max(cfg.train.microbatch_originals, 1)
    task = cfg.train.task
    out: dict[str, float] = {}

    if task in ("behavior", "joint"):
        keys = ("behavior_total", "L_robust", "L_pair_drop", "L_max_drop")
        totals = dict.fromkeys(keys, 0.0)
        weights = dict.fromkeys(keys, 0.0)
        for chunk in _chunk(dataset.groups, micro):
            values = _behavior_terms(model, cfg, stats, chunk).as_dict()
            for key in keys:
                value = values[key]
                if value != value:  # 정의되지 않은 항은 분모에서도 제외합니다.
                    continue
                totals[key] += value * len(chunk)
                weights[key] += len(chunk)
        out.update(
            {
                key: (totals[key] / weights[key]) if weights[key] else float("nan")
                for key in keys
            }
        )

    if task in ("flow", "joint"):
        keys = ("total", "L_next", "L_within", "L_roll")
        totals = dict.fromkeys(keys, 0.0)
        weights = dict.fromkeys(keys, 0.0)
        samples = enumerate_pairs(dataset, list(range(dataset.n_blocks)))
        for bucket in group_by_cut(samples):
            for chunk in _chunk(bucket, micro):
                values = _flow_terms(model, cfg, stats, chunk).as_dict()
                for key in keys:
                    value = values[key]
                    if value != value:
                        continue
                    totals[key] += value * len(chunk)
                    weights[key] += len(chunk)
        flow = {
            key: (totals[key] / weights[key]) if weights[key] else float("nan") for key in keys
        }
        out["flow_total"] = flow.pop("total")
        out.update(flow)

    # joint objective 값. flow-only task에서는 flow total이 곧 objective입니다.
    behavior_total = out.get("behavior_total", float("nan"))
    flow_total = out.get("flow_total", float("nan"))
    if task == "flow":
        out["total"] = flow_total
    elif task == "behavior":
        out["total"] = behavior_total
    else:
        # 한쪽이 정의되지 않으면 정의된 항만 씁니다.
        parts = [x for x in (behavior_total, cfg.train.flow_weight * flow_total) if x == x]
        out["total"] = sum(parts) if parts else float("nan")
    for key in ("L_next", "L_within", "L_roll", "behavior_total", "L_robust", "L_pair_drop",
                "L_max_drop", "flow_total"):
        out.setdefault(key, float("nan"))
    return out


# 하위 호환 이름 (flow-only 경로에서 쓰던 이름).
validation_loss = validation_metrics


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
    label_policy = _check_label_policy(datasets)
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
        "label_policy_hash": label_policy,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "task": cfg.train.task,
        "model_name": cfg.model.name,
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

    task = cfg.train.task
    select_metric = cfg.train.resolved_select_metric()
    supervision_seen = {"pair_drop": False, "robust": False, "max_drop": False}

    for epoch in range(start_epoch, cfg.train.max_epochs):
        started = time.time()
        model.train()
        order = torch.randperm(n_groups, generator=sampler).tolist()
        epoch_total, epoch_weight = 0.0, 0.0
        for group_chunk in _chunk(order, batch_size):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            groups = [train_set.groups[i] for i in group_chunk]

            # ---- behavior view ----
            behavior_chunks = _chunk(groups, micro) if task in ("behavior", "joint") else []
            n_behavior = sum(len(chunk) for chunk in behavior_chunks)
            # ---- flow view ----
            flow_buckets: list[list] = []
            if task in ("flow", "joint"):
                samples = sample_pairs(train_set, sampler, cfg.train.cuts_per_pair, group_chunk)
                flow_buckets = [
                    chunk for bucket in group_by_cut(samples) for chunk in _chunk(bucket, micro)
                ]
            n_flow = sum(len(chunk) for chunk in flow_buckets)

            batch_loss = 0.0
            for chunk in behavior_chunks:
                terms = _behavior_terms(model, cfg, stats, chunk)
                for name, term in (
                    ("pair_drop", terms.pair_drop),
                    ("robust", terms.robust),
                    ("max_drop", terms.max_drop),
                ):
                    if term.defined:
                        supervision_seen[name] = True
                scaled = terms.total * (len(chunk) / max(n_behavior, 1))
                if optimizer is not None and scaled.requires_grad:
                    scaled.backward()
                batch_loss += float(terms.total) * (len(chunk) / max(n_behavior, 1))
            for chunk in flow_buckets:
                terms = _flow_terms(model, cfg, stats, chunk)
                weight = (len(chunk) / max(n_flow, 1)) * (
                    cfg.train.flow_weight if task == "joint" else 1.0
                )
                scaled = terms.total * weight
                if optimizer is not None and scaled.requires_grad:
                    scaled.backward()
                batch_loss += float(terms.total) * weight
            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                optimizer.step()
            epoch_total += batch_loss * len(group_chunk)
            epoch_weight += len(group_chunk)

        # 실제 label로 학습된 head만 표시합니다.
        if hasattr(model, "mark_trained"):
            for name, seen in supervision_seen.items():
                if seen:
                    model.mark_trained(name)

        val = validation_metrics(model, val_set, stats, cfg)
        if select_metric not in val:
            raise ValueError(
                f"train.select_metric={select_metric!r} is not a validation key; "
                f"available: {sorted(val)}"
            )
        record = EpochRecord(
            epoch=epoch,
            train_total=epoch_total / max(epoch_weight, 1.0),
            val_total=val["total"],
            val_next=val["L_next"],
            val_within=val["L_within"],
            val_roll=val["L_roll"],
            seconds=time.time() - started,
            val_behavior_total=val["behavior_total"],
            val_robust=val["L_robust"],
            val_pair_drop=val["L_pair_drop"],
            val_max_drop=val["L_max_drop"],
            val_flow_total=val["flow_total"],
        )
        history.append(record)

        selected = val[select_metric]
        if selected != selected:  # NaN이면 개선으로 보지 않습니다.
            improved = False
        else:
            improved = selected < best_val - 1e-9
        payload_args = (model, optimizer, stats, cfg, hashes, epoch)
        if improved:
            best_val, best_epoch = selected, epoch
            patience_left = cfg.train.patience
            atomic_save(
                _checkpoint_payload(
                    *payload_args, best_val, best_epoch, history, sampler, meta
                ),
                run_dir / CHECKPOINT_BEST,
            )
        else:
            patience_left -= 1
        atomic_save(
            _checkpoint_payload(*payload_args, best_val, best_epoch, history, sampler, meta),
            last_path,
        )
        if optimizer is None:
            break  # parameter가 없는 baseline은 한 번의 평가로 끝냅니다.
        if patience_left <= 0:
            break

    trained_heads = (
        model.trained_heads() if hasattr(model, "trained_heads") else {}
    )
    summary = {
        "run_id": cfg.run.run_id,
        "model": cfg.model.name,
        "task": task,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "select_metric": select_metric,
        "seed": cfg.run.seed,
        "params": report.as_dict(),
        "hashes": hashes,
        "best_epoch": best_epoch,
        "best_val_total": None if best_val == float("inf") else best_val,
        "epochs_run": len(history),
        "history": [record.__dict__ for record in history],
        "n_train_originals": len(train_set.groups),
        "n_validation_originals": len(val_set.groups),
        "trained_heads": trained_heads,
        "behavior_supervision_seen": supervision_seen,
        "label_coverage": _label_coverage(train_set),
    }
    if task in ("behavior", "joint") and not any(supervision_seen.values()):
        # behavior supervision이 전혀 없으면 joint 성공으로 보고하지 않습니다.
        summary["warning"] = (
            "no behavior supervision was available in this run; "
            "this is not a joint training result"
        )
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
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task": cfg.train.task,
        "model_name": cfg.model.name,
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
    for key in (
        "config_hash",
        "stats_hash",
        "train_subset_hash",
        "label_policy_hash",
        "task",
        "model_name",
    ):
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
    version = payload.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema version {version!r} != expected "
            f"{CHECKPOINT_SCHEMA_VERSION!r}. v1 flow-only checkpoints are not reinterpreted "
            "as v2 supervised models; retrain, or read them with the aimo v1 revision"
        )
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


def checkpoint_info(payload: dict) -> dict:
    """checkpoint의 schema/model/task version과 head 학습 여부를 요약합니다."""
    heads = payload.get("model", {}).get("head_trained")
    from .behavior import HEAD_NAMES

    trained = {}
    if heads is not None:
        trained = {
            name: bool(float(heads[i]) > 0) for i, name in enumerate(HEAD_NAMES)
        }
    return {
        "schema_version": payload.get("schema_version"),
        "task": payload.get("task"),
        "model_name": payload.get("model_name"),
        "epoch": payload.get("epoch"),
        "best_epoch": payload.get("best_epoch"),
        "trained_heads": trained,
        "hashes": payload.get("hashes", {}),
    }
