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

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy
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
from .losses import (
    behavior_loss,
    behavior_term_counts,
    compute_loss,
    flow_term_counts,
)
from .model import build_model
from .runtime import atomic_save, atomic_write_json, guard_config, resolve_device

CHECKPOINT_LAST = "last.pt"
CHECKPOINT_BEST = "best.pt"

# checkpoint schema 버전. v1(flow-only) checkpoint를 새 모델로 조용히 해석하지 않습니다.
# v2 -> v3: 역할별 seed, content-sensitive data hash, Python/NumPy RNG를 포함합니다.
# v2 이하 checkpoint는 명시적으로 거부하며 조용히 재해석하지 않습니다.
CHECKPOINT_SCHEMA_VERSION = "aimo-checkpoint-v3"
COMPATIBLE_CHECKPOINT_VERSIONS = (CHECKPOINT_SCHEMA_VERSION,)


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


# validation 지표가 필요로 하는 label 항.
METRIC_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "L_robust": ("robust",),
    "L_pair_drop": ("pair_drop",),
    "L_max_drop": ("max_drop",),
    "behavior_total": ("robust", "pair_drop", "max_drop"),
    "L_next": ("next",),
    "L_within": ("within",),
    "L_roll": ("roll",),
    "flow_total": ("next", "within", "roll"),
}


def _split_label_counts(
    dataset: PageDataset, cfg: Config, stats: NormStats
) -> dict[str, int]:
    """model forward 없이 split 전체의 항별 유효 원문 수를 셉니다."""
    micro = max(cfg.train.microbatch_originals, 1)
    counts: dict[str, int] = {}
    if cfg.train.task in ("behavior", "joint"):
        for chunk in _chunk(dataset.groups, micro):
            _accumulate_counts(
                counts,
                behavior_term_counts(
                    collate_panels(chunk), use_max_drop=cfg.train.use_max_drop
                ),
            )
    if cfg.train.task in ("flow", "joint"):
        samples = enumerate_pairs(dataset, list(range(dataset.n_blocks)))
        for bucket in group_by_cut(samples):
            for chunk in _chunk(bucket, micro):
                _accumulate_counts(
                    counts, flow_term_counts(collate(chunk), stats, tuple(cfg.train.horizons))
                )
    return counts


def check_supervision(
    cfg: Config, train_set: PageDataset, val_set: PageDataset, stats: NormStats
) -> dict:
    """학습 시작 전에 supervision과 선택 지표의 유효 label을 확인합니다.

    - 활성화한 behavior objective에 supervision이 전혀 없으면 fail-fast합니다.
    - 선택된 validation 지표에 유효 label이 하나도 없으면 fail-fast합니다.
    """
    train_counts = _split_label_counts(train_set, cfg, stats)
    val_counts = _split_label_counts(val_set, cfg, stats)
    task = cfg.train.task
    enabled = []
    if task in ("behavior", "joint"):
        if cfg.train.w_pair_drop > 0:
            enabled.append("pair_drop")
        if cfg.train.w_robust > 0:
            enabled.append("robust")
        if cfg.train.use_max_drop and cfg.train.w_max_drop > 0:
            enabled.append("max_drop")
        if enabled and not any(train_counts.get(name, 0) > 0 for name in enabled):
            raise ValueError(
                f"train.task={task!r} enables behavior objectives {enabled} but the train split "
                f"has no valid behavior label (counts={train_counts}). Run 'aimo build-labels' "
                "first, or select train.task='flow' explicitly for a flow-only run"
            )
    selected = cfg.train.resolved_select_metric()
    required = METRIC_REQUIREMENTS.get(selected)
    if required is None and selected == "total":
        required = (
            ("next", "within", "roll")
            if task == "flow"
            else ("robust", "pair_drop", "max_drop")
        )
    if required and not any(val_counts.get(name, 0) > 0 for name in required):
        raise ValueError(
            f"train.select_metric={selected!r} needs at least one of {list(required)} on the "
            f"validation split but none has a valid label (counts={val_counts}); "
            "pick a metric that your labels actually support before starting"
        )
    return {
        "train_label_counts": train_counts,
        "validation_label_counts": val_counts,
        "enabled_behavior_objectives": enabled,
        "select_metric": selected,
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


def _behavior_terms(model, cfg: Config, stats: NormStats, batch, denominators=None):
    """behavior loss를 한 microbatch에 대해 계산합니다 (global denominator 적용)."""
    return behavior_loss(
        model,
        batch,
        stats,
        w_robust=cfg.train.w_robust,
        w_pair_drop=cfg.train.w_pair_drop,
        w_max_drop=cfg.train.w_max_drop,
        use_max_drop=cfg.train.use_max_drop,
        denominators=denominators,
    )


def _flow_terms(model, cfg: Config, stats: NormStats, batch, denominators=None):
    """flow loss를 한 microbatch(단일 cut)에 대해 계산합니다 (global denominator 적용)."""
    return compute_loss(
        model,
        batch,
        stats,
        w_next=cfg.train.w_next,
        w_within=cfg.train.w_within,
        w_roll=cfg.train.w_roll,
        horizons=tuple(cfg.train.horizons),
        denominators=denominators,
    )


def _accumulate_counts(target: dict[str, int], addition: dict[str, int]) -> dict[str, int]:
    for key, value in addition.items():
        target[key] = target.get(key, 0) + int(value)
    return target


@torch.no_grad()
def validation_metrics(
    model: nn.Module, dataset: PageDataset, stats: NormStats, cfg: Config
) -> dict[str, float]:
    """결정적 validation 지표. 학습과 **같은 sum/count 방식**으로 집계합니다.

    behavior는 panel 순서대로, flow는 모든 cut과 sibling 조합을 고정 순서로 나열합니다.
    각 loss 항은 자기 항의 유효 원문 수로만 나눕니다 (서로 다른 항의 count를 합치지
    않습니다). 따라서 microbatch 크기가 값을 바꾸지 않습니다.
    """
    model.eval()
    micro = max(cfg.train.microbatch_originals, 1)
    task = cfg.train.task
    out: dict[str, float] = {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    def add(name: str, term) -> None:
        sums[name] = sums.get(name, 0.0) + float(term.total)
        counts[name] = counts.get(name, 0) + term.count

    def mean(name: str) -> float:
        return sums[name] / counts[name] if counts.get(name) else float("nan")

    if task in ("behavior", "joint"):
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else (
            stats.input_state_scale.device
        )
        for chunk in _chunk(dataset.groups, micro):
            terms = _behavior_terms(model, cfg, stats, collate_panels(chunk).to(device))
            add("L_robust", terms.robust)
            add("L_pair_drop", terms.pair_drop)
            add("L_max_drop", terms.max_drop)
        out["L_robust"] = mean("L_robust")
        out["L_pair_drop"] = mean("L_pair_drop")
        out["L_max_drop"] = mean("L_max_drop")
        parts = []
        for name, weight in (
            ("L_robust", cfg.train.w_robust),
            ("L_pair_drop", cfg.train.w_pair_drop),
            ("L_max_drop", cfg.train.w_max_drop),
        ):
            if counts.get(name):
                parts.append(weight * out[name])
        out["behavior_total"] = sum(parts) if parts else float("nan")

    if task in ("flow", "joint"):
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else (
            stats.input_state_scale.device
        )
        samples = enumerate_pairs(dataset, list(range(dataset.n_blocks)))
        for bucket in group_by_cut(samples):
            for chunk in _chunk(bucket, micro):
                terms = _flow_terms(model, cfg, stats, collate(chunk).to(device))
                add("L_next", terms.next_update)
                add("L_within", terms.within)
                add("L_roll", terms.roll)
        out["L_next"] = mean("L_next")
        out["L_within"] = mean("L_within")
        out["L_roll"] = mean("L_roll")
        parts = []
        for name, weight in (
            ("L_next", cfg.train.w_next),
            ("L_within", cfg.train.w_within),
            ("L_roll", cfg.train.w_roll),
        ):
            if counts.get(name):
                parts.append(weight * out[name])
        out["flow_total"] = sum(parts) if parts else float("nan")

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
    out["n_valid"] = float(sum(counts.values()))
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

    seeds = cfg.run.resolved_seeds()
    # 학습 초기화와 dropout은 train seed만 씁니다.
    torch.manual_seed(seeds["train"])
    random.seed(seeds["train"])
    numpy.random.seed(seeds["train"] % (2**32))
    label_policy = _check_label_policy(datasets)
    train_set = datasets["train"].subset(cfg.data.subset_fraction)
    val_set = datasets["validation"]
    # normalization은 각 training subset 안에서만 계산합니다.
    stats = compute_norm_stats(train_set, floor=cfg.data.scale_floor)

    # device는 config에서 결정하고, 없으면 조용히 CPU로 내려가지 않고 오류를 냅니다.
    device = resolve_device(cfg.run.device)
    model = build_model(cfg, train_set.hidden_size, train_set.n_blocks, train_set.n_landmarks)
    model.to(device)
    stats = stats.to(device)
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
        # data/split/label hash는 train seed와 무관해야 합니다.
        "data_seed": seeds["data"],
        "split_seed": seeds["split"],
        "data_schema_version": DATA_SCHEMA_VERSION,
        "task": cfg.train.task,
        "model_name": cfg.model.name,
    }

    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    history: list[EpochRecord] = []
    sampler = torch.Generator().manual_seed(seeds["sampler"] + 101)

    last_path = run_dir / CHECKPOINT_LAST
    if resume and last_path.exists():
        payload = torch.load(last_path, map_location="cpu")
        _check_hashes(payload["hashes"], hashes)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        _restore_rng_state(payload.get("rng"), payload)
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
    # 학습 시작 전에 supervision과 선택 지표의 유효 label을 확인합니다.
    supervision_report = check_supervision(cfg, train_set, val_set, stats)
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

            # ---- microbatch를 먼저 만들고 항별 global denominator를 셉니다 ----
            behavior_batches = []
            behavior_denoms: dict[str, int] = {}
            if task in ("behavior", "joint"):
                for chunk in _chunk(groups, micro):
                    batch = collate_panels(chunk)
                    behavior_batches.append(batch)
                    _accumulate_counts(
                        behavior_denoms,
                        behavior_term_counts(batch, use_max_drop=cfg.train.use_max_drop),
                    )
            flow_batches = []
            flow_denoms: dict[str, int] = {}
            if task in ("flow", "joint"):
                samples = sample_pairs(train_set, sampler, cfg.train.cuts_per_pair, group_chunk)
                for bucket in group_by_cut(samples):
                    for chunk in _chunk(bucket, micro):
                        batch = collate(chunk)
                        flow_batches.append(batch)
                        _accumulate_counts(
                            flow_denoms,
                            flow_term_counts(batch, stats, tuple(cfg.train.horizons)),
                        )

            # ---- 같은 global denominator로 나눈 항을 microbatch마다 누적합니다 ----
            batch_loss = 0.0
            for batch in behavior_batches:
                # microbatch만 device로 옮깁니다 (전체 dataset을 올리지 않습니다).
                terms = _behavior_terms(
                    model, cfg, stats, batch.to(device), denominators=behavior_denoms
                )
                # 가중치가 0인 항은 objective에 기여하지 않으므로 학습된 것으로 보지 않습니다.
                for name, term, weight in (
                    ("pair_drop", terms.pair_drop, cfg.train.w_pair_drop),
                    ("robust", terms.robust, cfg.train.w_robust),
                    ("max_drop", terms.max_drop, cfg.train.w_max_drop),
                ):
                    if term.defined and weight > 0:
                        supervision_seen[name] = True
                if optimizer is not None and terms.total.requires_grad:
                    terms.total.backward()
                batch_loss += float(terms.total)
            for batch in flow_batches:
                terms = _flow_terms(
                    model, cfg, stats, batch.to(device), denominators=flow_denoms
                )
                weight = cfg.train.flow_weight if task == "joint" else 1.0
                scaled = terms.total * weight
                if optimizer is not None and scaled.requires_grad:
                    scaled.backward()
                batch_loss += float(scaled)
            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                optimizer.step()
            epoch_total += batch_loss
            epoch_weight += 1.0

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
        "seeds": seeds,
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
        "supervision": supervision_report,
        "label_coverage": _label_coverage(train_set),
        # flow-only 실행은 명시적으로 고른 경우이며 joint 결과가 아닙니다.
        "is_joint_result": task == "joint" and any(supervision_seen.values()),
    }
    if task == "flow":
        summary["note"] = (
            "flow-only run selected explicitly (train.task='flow'); this is not a joint "
            "training result"
        )
    if task in ("behavior", "joint") and not any(supervision_seen.values()):
        # behavior supervision이 전혀 없으면 joint 성공으로 보고하지 않습니다.
        summary["warning"] = (
            "no behavior supervision was available in this run; "
            "this is not a joint training result"
        )
    atomic_write_json(run_dir / "train_summary.json", summary)
    return summary


def _rng_state() -> dict:
    """resume 재현성에 필요한 RNG 상태.

    CPU(torch), Python `random`, NumPy를 보존합니다. CUDA 상태도 사용 가능할 때 저장하지만
    **CUDA 재현성은 로컬에서 검증하지 않았습니다 (SERVER_PENDING).**
    """
    state = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "cuda": None,
        "cuda_verified": False,
    }
    if torch.cuda.is_available():  # pragma: no cover - 로컬 CPU에서는 실행되지 않습니다
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None, payload: dict) -> None:
    """저장된 RNG 상태를 복원합니다. 구버전 key도 읽습니다."""
    if not state:
        torch.set_rng_state(payload["torch_rng_state"])
        return
    torch.set_rng_state(state["torch_cpu"])
    if state.get("python") is not None:
        random.setstate(
            tuple(
                tuple(item) if isinstance(item, list) else item for item in state["python"]
            )
        )
    if state.get("numpy") is not None:
        numpy.random.set_state(tuple(state["numpy"]))
    if state.get("cuda") is not None and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all(state["cuda"])


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
        "rng": _rng_state(),
        "torch_rng_state": torch.get_rng_state(),  # 하위 호환 key
        "sampler_rng_state": sampler.get_state(),
        "format_version": 3,
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
    if version not in COMPATIBLE_CHECKPOINT_VERSIONS:
        raise ValueError(
            f"checkpoint schema version {version!r} is not compatible with "
            f"{list(COMPATIBLE_CHECKPOINT_VERSIONS)}. Earlier checkpoints (no version, or "
            "aimo-checkpoint-v2) are not reinterpreted: v2 used a per-ID data hash and did not "
            "store role-separated seeds or Python/NumPy RNG. Retrain, or read them with the "
            "revision that wrote them"
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
