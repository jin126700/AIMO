"""LRT-v1 실험: transport loss, baselines, support-swap control, CPU toy, macro audit.

이 module은 CPU에서 architecture plumbing을 검증하는 것까지만 합니다. 실제 FP32 Page
실험과 GPU 서버 실행은 다음 단계입니다 (SERVER-UNTESTED).

Primary metric은 **transport ratio**입니다.

    E_pred = mean_valid_scalar ||ΔU_norm - ΔU_hat_norm||²
    E_zero = mean_valid_scalar ||ΔU_norm||²
    R      = E_pred / max(E_zero, τ)

Zero predictor는 R = 1로 해석합니다. variants/folds를 해당 original 안에서 먼저 평균한 뒤
originals를 동일 가중치로 평균합니다 (original-balanced).

    L = L_transport + 0.1 * L_consistency
    L_consistency = 1 - cosine(z1, z2)   (독립 support cell dropout 2회)

behavior loss / Flow next loss / energy loss / InfoNCE / negative contrastive /
recipe classification은 넣지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .lrt import (
    LANDMARK_ALL_COMMON,
    LANDMARK_MODES,
    LRT_SCHEMA,
    LRTModel,
    MacroNormStats,
    build_pair_tensors,
    fit_macro_norm_stats,
    landmark_mask,
    query_folds,
    support_macros_for,
)
from .macro_page import (
    DEFAULT_N_MACRO,
    MacroPage,
    common_valid,
    macro_relation,
    relation_energy,
    to_macro_page,
)
from .runtime import atomic_save, atomic_write_json

LRT_CHECKPOINT_SCHEMA = "aimo-lrt-v1"
# transport ratio 판정 문구. server 결과 없이 validated/successful이라고 쓰지 않습니다.
STATUS_CPU_VALIDATED = "IMPLEMENTED / CPU-VALIDATED / SERVER-UNTESTED"


# --------------------------------------------------------------------------------------
# transport loss
# --------------------------------------------------------------------------------------


@dataclass
class TransportTerm:
    """한 (pair, fold)의 transport 결과."""

    e_pred: float
    e_zero: float
    ratio: float
    n_cells: int
    below_floor: bool = False
    n_scalar: int = 0

    @property
    def mse_pred(self):
        return self.e_pred

    @property
    def mse_zero(self):
        return self.e_zero


def denominator(e_zero: float, floor: float) -> float:
    """max(E_zero, τ). floor는 config에서 명시적으로 와야 합니다."""
    return max(e_zero, floor)


def transport_term(
    delta_norm: Tensor, delta_hat: Tensor, valid: Tensor, floor: float
) -> TransportTerm:
    """valid landmark/stream cell에서 E_pred, E_zero, ratio를 계산합니다.

    delta_norm / delta_hat : [Q, P_sel, 2, H]
    valid                  : [P_sel] bool
    """
    if floor <= 0 or not bool(valid.any()):
        raise ValueError("positive floor and nonempty valid mask required")
    n_scalar = int(valid.sum()) * delta_norm.shape[0] * delta_norm.shape[2] * delta_norm.shape[3]
    mask = valid.view(1, -1, 1, 1).to(device=delta_norm.device, dtype=delta_norm.dtype)
    e_pred = float(((delta_norm - delta_hat).pow(2) * mask).sum() / n_scalar)
    e_zero = float((delta_norm.pow(2) * mask).sum() / n_scalar)
    den = denominator(e_zero, floor)
    return TransportTerm(
        e_pred=e_pred,
        e_zero=e_zero,
        ratio=e_pred / den,
        n_cells=int(valid.sum()) * delta_norm.shape[0] * delta_norm.shape[2],
        below_floor=e_zero < floor,
        n_scalar=n_scalar,
    )


def transport_loss(
    delta_norm: Tensor, delta_hat: Tensor, valid: Tensor, floor: float
) -> Tensor:
    """gradient가 흐르는 transport ratio (한 fold). denominator는 상수로 둡니다."""
    if floor <= 0 or not bool(valid.any()):
        raise ValueError("positive floor and nonempty valid mask required")
    n_scalar = int(valid.sum()) * delta_norm.shape[0] * delta_norm.shape[2] * delta_norm.shape[3]
    mask = valid.view(1, -1, 1, 1).to(device=delta_norm.device, dtype=delta_norm.dtype)
    e_pred = ((delta_norm - delta_hat).pow(2) * mask).sum() / n_scalar
    e_zero = float((delta_norm.pow(2) * mask).sum() / n_scalar)
    return e_pred / denominator(e_zero, floor)


def original_balanced_mean(per_original: dict[str, list[float]]) -> float:
    """original 안에서 먼저 평균한 뒤 originals를 동일 가중치로 평균합니다."""
    means = [sum(values) / len(values) for values in per_original.values() if values]
    return sum(means) / len(means) if means else float("nan")


def consistency_loss(z1: Tensor, z2: Tensor) -> Tensor:
    """1 - cosine(z1, z2)."""
    return 1.0 - torch.nn.functional.cosine_similarity(z1, z2, dim=-1).mean()


def support_dropout_mask(
    n_cells: int, rate: float, generator: torch.Generator, device: torch.device
) -> Tensor:
    """[1, n_cells] bool mask. support cell이 전부 막히지 않도록 보장합니다."""
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"support_dropout must be in [0, 1), got {rate}")
    keep = torch.rand(n_cells, generator=generator) >= rate
    if not bool(keep.any()):
        keep[int(torch.randint(0, n_cells, (1,), generator=generator))] = True
    return keep.to(device).unsqueeze(0)


# --------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------


class ZeroTransport:
    """ΔU_hat = 0. transport ratio는 정의상 1입니다 (denominator floor 위에서)."""

    name = "zero"

    def predict(self, tensors: dict, query_macros, landmarks) -> Tensor:
        target = tensors["delta_norm"][:, list(query_macros)][:, :, landmarks]
        return torch.zeros_like(target)


class TrainMeanTransport:
    """train originals에서만 계산한 macro × stream relation mean.

    train pair의 normalized ΔU를 macro/stream별로 평균합니다. held-out 통계를 쓰지 않습니다.
    """

    name = "train_mean"

    def __init__(self, mean: Tensor) -> None:
        self.mean = mean  # [G, 2, H]

    @classmethod
    def fit(
        cls, pairs: list[tuple[MacroPage, MacroPage]], stats: MacroNormStats
    ) -> TrainMeanTransport:
        total = None
        count = 0
        for original, variant in pairs:
            delta = stats.norm_update(macro_relation(original, variant)[None])[0]
            valid = common_valid(original, variant)
            masked = delta[:, valid]  # [G, Pv, 2, H]
            contribution = masked.mean(dim=1)  # [G, 2, H]
            total = contribution if total is None else total + contribution
            count += 1
        if total is None:
            raise ValueError("train-mean baseline needs at least one train pair")
        return cls(total / count)

    def predict(self, tensors: dict, query_macros, landmarks) -> Tensor:
        n_p = int(landmarks.numel())
        selected = self.mean[list(query_macros)]  # [Q, 2, H]
        return selected.unsqueeze(1).expand(-1, n_p, -1, -1).unsqueeze(0).clone()


class Rank4LinearTransport:
    """rank4_linear_transport: train-only rank-4 native relation basis 기반 선형 transport.

    **historical U4의 재현이 아닙니다.** repo에 historical U4 구현이 없어서 추측 재현을
    하지 않았고, 아래 정의로 새로 만든 baseline입니다.

    정의:
      1. train pair의 normalized ΔU를 [n_samples, H]로 모아 stream별로 rank-4 orthonormal
         basis `B_c` (H×4)를 SVD로 구합니다 (train originals의 pair만 사용).
      2. support macro의 relation을 basis에 투영해 pair coefficient
         `a_c = mean_{support, landmark} B_c^T ΔU_norm[g, p, c]` (4-dim)를 만듭니다.
      3. query site 예측은 `ΔU_hat[g, p, c] = B_c a_c` 입니다 (site 독립, state 독립).

    fit 범위는 train pair뿐이고 held-out 통계를 쓰지 않습니다.
    """

    name = "rank4_linear_transport"
    rank = 4

    def __init__(self, basis: dict[int, Tensor]) -> None:
        self.basis = basis  # {stream: [H, 4]}

    @classmethod
    def fit(
        cls, pairs: list[tuple[MacroPage, MacroPage]], stats: MacroNormStats
    ) -> Rank4LinearTransport:
        if not pairs:
            raise ValueError("rank4 baseline needs train pairs")
        hidden = pairs[0][0].hidden_size
        grams = {c: torch.zeros(hidden, hidden, dtype=torch.float64) for c in (0, 1)}
        for original, variant in pairs:
            delta = stats.norm_update(macro_relation(original, variant)[None])[0]
            valid = common_valid(original, variant)
            for c in (0, 1):
                matrix = delta[:, valid, c].reshape(-1, hidden).double()
                grams[c].addmm_(matrix.T, matrix)
        basis = {}
        for c, gram in grams.items():
            _, vectors = torch.linalg.eigh(gram)
            basis[c] = vectors[:, -cls.rank:].float().contiguous()
        return cls(basis)

    def predict(self, tensors: dict, query_macros, landmarks) -> Tensor:
        delta = tensors["delta_norm"][0]  # [G, P, 2, H]
        n_macro = delta.shape[0]
        support = support_macros_for(n_macro, tuple(query_macros))
        n_p = int(landmarks.numel())
        outputs = []
        for stream in (0, 1):
            observed = delta[support][:, landmarks, stream].reshape(-1, delta.shape[-1])
            coefficients = (observed @ self.basis[stream]).mean(dim=0)  # [4]
            outputs.append(self.basis[stream] @ coefficients)  # [H]
        stacked = torch.stack(outputs)  # [2, H]
        return (
            stacked.view(1, 1, 1, 2, -1)
            .expand(1, len(query_macros), n_p, 2, stacked.shape[-1])
            .clone()
        )


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------


def swap_donor_index(index: int, original_ids: list[str], eval_seed: int) -> int:
    """support-swap donor. 같은 pair / 같은 original을 금지합니다.

    outcome/label을 보고 고르지 않고 eval seed로 deterministic하게 정합니다.
    """
    n = len(original_ids)
    if n < 2:
        raise ValueError("support-swap needs at least two pairs")
    cross = [j for j in range(n) if original_ids[j] != original_ids[index]]
    if not cross:
        raise ValueError("support-swap requires a cross-original donor")
    pool = cross
    digest = hashlib.sha256(f"{eval_seed}:{index}:{original_ids[index]}".encode()).hexdigest()
    return pool[int(digest, 16) % len(pool)]


@torch.no_grad()
def evaluate_transport(
    model: LRTModel | None,
    pairs: list[tuple[MacroPage, MacroPage]],
    stats: MacroNormStats,
    *,
    floor: float,
    landmark_mode: str = LANDMARK_ALL_COMMON,
    baseline=None,
    support_swap: bool = False,
    eval_seed: int = 0,
) -> dict:
    """transport ratio와 support-swap gap을 original-balanced로 집계합니다."""
    if landmark_mode not in LANDMARK_MODES:
        raise ValueError(f"unknown landmark_mode {landmark_mode!r}")
    if model is not None:
        model.eval()
    device = next(model.parameters()).device if model is not None else stats.update_scale.device
    stats = stats.to(device)
    tensors = [build_pair_tensors(o, v, stats) for o, v in pairs]
    records = []
    original_ids = [o.original_id for o, _ in pairs]
    per_original: dict[str, list[float]] = {}
    swap_per_original: dict[str, list[float]] = {}
    n_below_floor = 0
    n_folds = 0
    n_macro = pairs[0][0].n_macro

    for index, ((original, variant), payload) in enumerate(zip(pairs, tensors, strict=True)):
        mask = landmark_mask(original, variant, landmark_mode)
        if not bool(mask.any()):
            continue  # 공통 valid landmark가 없으면 zero embedding을 끼우지 않고 건너뜁니다
        landmarks = torch.nonzero(mask).flatten()
        valid = mask[landmarks]
        for fold in query_folds(n_macro):
            support = support_macros_for(n_macro, fold)
            target = payload["delta_norm"][:, list(fold)][:, :, landmarks][0]
            if baseline is not None:
                prediction = baseline.predict(payload, fold, landmarks)[0]
            else:
                prediction = model(
                    payload["delta_norm"],
                    payload["original_state_norm"],
                    payload["original_update_norm"],
                    support,
                    fold,
                    landmarks,
                    payload["relative_positions"],
                ).delta_hat[0]
            term = transport_term(target, prediction, valid, floor)
            records.append({"original_id": original.original_id, "variant_id": variant.variant_id,
                            "fold": list(fold), "mse_pred": term.mse_pred, "mse_zero": term.mse_zero,
                            "ratio": term.ratio, "below_floor": term.below_floor,
                            "n_scalar": term.n_scalar, "n_cells": term.n_cells})
            per_original.setdefault(original.original_id, []).append(term.ratio)
            n_below_floor += int(term.below_floor)
            n_folds += 1
            if support_swap and model is not None:
                donor = swap_donor_index(index, original_ids, eval_seed)
                donor_payload = tensors[donor]
                donor_original, donor_variant = pairs[donor]
                donor_mask = landmark_mask(donor_original, donor_variant, landmark_mode)
                if not bool(donor_mask.any()):
                    continue
                donor_landmarks = torch.nonzero(donor_mask).flatten()
                z_donor, _ = model.encode(
                    donor_payload["delta_norm"],
                    support_macros_for(n_macro, fold),
                    donor_landmarks,
                    donor_payload["relative_positions"],
                )
                swapped = model.decode(
                    z_donor,
                    payload["original_state_norm"],
                    payload["original_update_norm"],
                    fold,
                    landmarks,
                    payload["relative_positions"],
                )[0]
                swap_term = transport_term(target, swapped, valid, floor)
                records[-1].update(swap_ratio=swap_term.ratio, donor_original_id=original_ids[donor])
                swap_per_original.setdefault(original.original_id, []).append(swap_term.ratio)

    result = {
        "per_original": {k: sum(v)/len(v) for k, v in per_original.items()},
        "records": records,
        "landmark_mode": landmark_mode,
        "transport_ratio": original_balanced_mean(per_original),
        "n_originals": len(per_original),
        "n_folds": n_folds,
        "n_below_denominator_floor": n_below_floor,
        "denominator_floor": floor,
    }
    if support_swap and swap_per_original:
        swap_ratio = original_balanced_mean(swap_per_original)
        result["swap_per_original"] = {k: sum(v)/len(v) for k, v in swap_per_original.items()}
        result["swap_transport_ratio"] = swap_ratio
        result["swap_gap"] = swap_ratio - result["transport_ratio"]
        result["swap_note"] = (
            "swap_gap > 0은 relation representation을 실제로 사용한다는 evidence이며 "
            "causal evidence가 아닙니다."
        )
    return result


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------


def train_lrt(
    pairs: list[tuple[MacroPage, MacroPage]],
    validation_pairs: list[tuple[MacroPage, MacroPage]],
    stats: MacroNormStats,
    lrt_cfg,
    *,
    floor: float,
    max_epochs: int = 200,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    patience: int = 40,
    batch_originals: int = 2,
    seed: int = 0,
    device: torch.device | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """LRT-v1을 학습합니다. checkpoint 선택은 validation L_transport만 씁니다."""
    if not pairs:
        raise ValueError("LRT training needs at least one train pair")
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    sample = pairs[0][0]
    model = LRTModel(
        hidden_size=sample.hidden_size,
        n_macro=sample.n_macro,
        n_landmarks=sample.n_landmarks,
        adapter_dim=lrt_cfg.adapter_dim,
        d_model=lrt_cfg.d_model,
        n_heads=lrt_cfg.n_heads,
        ffn_dim=lrt_cfg.ffn_dim,
        dropout=lrt_cfg.dropout,
        n_loops=lrt_cfg.n_loops,
        relation_dim=lrt_cfg.relation_dim,
        decoder_rank=lrt_cfg.decoder_rank,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stats = stats.to(device)
    tensors = [
        {key: value.to(device) for key, value in build_pair_tensors(o, v, stats).items()}
        for o, v in pairs
    ]
    masks = [landmark_mask(o, v, lrt_cfg.landmark_mode) for o, v in pairs]
    n_macro = sample.n_macro
    generator = torch.Generator().manual_seed(seed + 17)

    history = []
    best = {"epoch": -1, "validation_transport_ratio": float("inf")}
    best_state = None
    patience_left = patience
    order_generator = torch.Generator().manual_seed(seed + 101)
    # original 단위로 batch를 만들어 batch마다 optimizer step을 밟습니다.
    by_original: dict[str, list[int]] = {}
    for index, (original, _variant) in enumerate(pairs):
        by_original.setdefault(original.original_id, []).append(index)
    original_ids = sorted(by_original)

    start_epoch = 0
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    last_path = checkpoint_dir / "last.pt" if checkpoint_dir else None
    if last_path is not None and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved["stats_hash"] != stats.hash() or saved["seed"] != seed or saved["floor"] != floor:
            raise ValueError("resume metadata mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_state = saved["history"], saved["best"], saved["best_state"]
        patience_left = saved["patience_left"]
        start_epoch = saved["epoch"] + 1
        generator.set_state(saved["generator"].cpu())
        order_generator.set_state(saved["order_generator"].cpu())
        torch.set_rng_state(saved["rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
    for epoch in range(start_epoch, max_epochs):
        if patience_left <= 0:
            break
        epoch_start = time.monotonic()
        model.train()
        permutation = torch.randperm(len(original_ids), generator=order_generator).tolist()
        shuffled = [original_ids[i] for i in permutation]
        epoch_loss, n_steps = 0.0, 0
        for start in range(0, len(shuffled), batch_originals):
            batch_ids = shuffled[start : start + batch_originals]
            per_original: dict[str, list[Tensor]] = {}
            for original_id in batch_ids:
                for index in by_original[original_id]:
                    payload, mask = tensors[index], masks[index]
                    if not bool(mask.any()):
                        continue
                    landmarks = torch.nonzero(mask).flatten().to(device)
                    valid = mask[landmarks.cpu()].to(device)
                    for fold in query_folds(n_macro):
                        support = support_macros_for(n_macro, fold)
                        n_cells = len(support) * int(landmarks.numel()) * 2
                        view1 = support_dropout_mask(
                            n_cells, lrt_cfg.support_dropout, generator, device
                        )
                        view2 = support_dropout_mask(
                            n_cells, lrt_cfg.support_dropout, generator, device
                        )
                        out1 = model(
                            payload["delta_norm"],
                            payload["original_state_norm"],
                            payload["original_update_norm"],
                            support,
                            fold,
                            landmarks,
                            payload["relative_positions"],
                            cell_mask=view1,
                        )
                        z2, _ = model.encode(
                            payload["delta_norm"],
                            support,
                            landmarks,
                            payload["relative_positions"],
                            cell_mask=view2,
                        )
                        target = payload["delta_norm"][:, list(fold)][:, :, landmarks][0]
                        loss = transport_loss(target, out1.delta_hat[0], valid, floor)
                        loss = loss + lrt_cfg.consistency_weight * consistency_loss(
                            out1.z_rel, z2
                        )
                        per_original.setdefault(original_id, []).append(loss)
            if not per_original:
                continue
            # original-balanced: variants/folds를 original 안에서 평균한 뒤 originals 평균.
            total = torch.stack(
                [torch.stack(values).mean() for values in per_original.values()]
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(total)):
                raise ValueError("nonfinite training loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(total)
            n_steps += 1
        if n_steps == 0:
            raise ValueError("no trainable pair had a common valid landmark")

        validation = evaluate_transport(
            model,
            validation_pairs or pairs,
            stats,
            floor=floor,
            landmark_mode=lrt_cfg.landmark_mode,
        )
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / n_steps,
            "steps": n_steps,
            "validation_transport_ratio": validation["transport_ratio"],
        }
        history.append(record)
        if validation["transport_ratio"] < best["validation_transport_ratio"] - 1e-9:
            best = dict(record)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "history": history, "best": best, "best_state": best_state,
                         "patience_left": patience_left, "epoch": epoch, "seed": seed,
                         "floor": floor, "stats_hash": stats.hash(),
                         "generator": generator.get_state(), "order_generator": order_generator.get_state(),
                         "rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}, last_path)
            atomic_write_json(checkpoint_dir / "progress.json", {**record, "epoch_seconds": time.monotonic()-epoch_start})
            print(f"seed={seed} epoch={epoch} dev_R={validation['transport_ratio']:.6f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "schema": LRT_CHECKPOINT_SCHEMA,
        "model": model,
        "history": history,
        "best": best,
        "epochs_run": len(history),
        "params": model.param_report().as_dict(),
        "stats_hash": stats.hash(),
        "status": "REAL_CUDA_TRAINED" if device.type == "cuda" else STATUS_CPU_VALIDATED,
    }


def save_lrt_checkpoint(path: str | Path, result: dict, lrt_cfg, stats: MacroNormStats) -> Path:
    """LRT 전용 schema로 저장합니다. old Flow/Behavior checkpoint와 섞이지 않습니다."""
    payload = {
        "schema_version": LRT_CHECKPOINT_SCHEMA,
        "model": result["model"].state_dict(),
        "macro_norm_stats": stats.state_dict(),
        "lrt_config": {
            "n_macro": lrt_cfg.n_macro,
            "adapter_dim": lrt_cfg.adapter_dim,
            "d_model": lrt_cfg.d_model,
            "n_heads": lrt_cfg.n_heads,
            "ffn_dim": lrt_cfg.ffn_dim,
            "dropout": lrt_cfg.dropout,
            "n_loops": lrt_cfg.n_loops,
            "relation_dim": lrt_cfg.relation_dim,
            "decoder_rank": lrt_cfg.decoder_rank,
            "consistency_weight": lrt_cfg.consistency_weight,
            "support_dropout": lrt_cfg.support_dropout,
            "denominator_floor": lrt_cfg.denominator_floor,
            "landmark_mode": lrt_cfg.landmark_mode,
        },
        "model_meta": {
            "hidden_size": result["model"].hidden_size,
            "n_macro": result["model"].n_macro,
            "n_landmarks": result["model"].n_landmarks,
        },
        "best": result["best"],
        "history": result["history"],
        "status": STATUS_CPU_VALIDATED,
    }
    return atomic_save(payload, path)


def load_lrt_checkpoint(path: str | Path) -> tuple[LRTModel, MacroNormStats, dict]:
    """LRT checkpoint를 읽습니다. 다른 schema는 거부합니다."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = payload.get("schema_version")
    if version != LRT_CHECKPOINT_SCHEMA:
        raise ValueError(
            f"checkpoint schema {version!r} is not {LRT_CHECKPOINT_SCHEMA!r}; Flow/Behavior "
            "checkpoints are not loadable as LRT models"
        )
    meta = payload["model_meta"]
    cfg = payload["lrt_config"]
    model = LRTModel(
        hidden_size=int(meta["hidden_size"]),
        n_macro=int(meta["n_macro"]),
        n_landmarks=int(meta["n_landmarks"]),
        adapter_dim=int(cfg["adapter_dim"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        ffn_dim=int(cfg["ffn_dim"]),
        dropout=float(cfg["dropout"]),
        n_loops=int(cfg["n_loops"]),
        relation_dim=int(cfg["relation_dim"]),
        decoder_rank=int(cfg["decoder_rank"]),
    )
    model.load_state_dict(payload["model"])
    model.eval()
    return model, MacroNormStats.from_state_dict(payload["macro_norm_stats"]), payload


# --------------------------------------------------------------------------------------
# CPU toy datasets
# --------------------------------------------------------------------------------------


def _toy_fine_page(
    n_blocks: int,
    n_landmarks: int,
    hidden: int,
    generator: torch.Generator,
    original_id: str,
    variant_id: str,
    *,
    extra_updates: Tensor | None = None,
) -> object:
    """residual identity를 정확히 만족하는 toy Fine Page."""
    from .page import Page

    updates = torch.randn(n_blocks, n_landmarks, 2, hidden, generator=generator) * 0.2
    if extra_updates is not None:
        updates = updates + extra_updates
    state = torch.zeros(n_blocks + 1, n_landmarks, hidden)
    state[0] = torch.randn(n_landmarks, hidden, generator=generator) * 0.5
    for depth in range(n_blocks):
        state[depth + 1] = state[depth] + updates[depth].sum(dim=1)
    return Page(
        state=state,
        updates=updates,
        valid=torch.ones(n_landmarks, dtype=torch.bool),
        token_offsets=torch.arange(n_landmarks, dtype=torch.long),
        relative_positions=torch.linspace(0.0, 1.0, n_landmarks),
        original_id=original_id,
        variant_id=variant_id,
        provenance={"source": "synthetic", "toy": True},
    )


def make_toy_rank4(
    n_originals: int = 6,
    n_variants: int = 2,
    n_blocks: int = 32,
    n_macro: int = DEFAULT_N_MACRO,
    n_landmarks: int = 4,
    hidden: int = 8,
    seed: int = 0,
) -> dict:
    """Toy A — fixed rank4 relation.

        ΔU = B z      (state 의존 없음)

    목적: rank4 / low-rank baseline 계약 검증.
    """
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(2, hidden, 4, generator=generator)
    basis = basis / basis.norm(dim=1, keepdim=True)
    pairs = []
    for index in range(n_originals):
        original_id = f"toyA-{index:03d}"
        original = _toy_fine_page(
            n_blocks, n_landmarks, hidden, generator, original_id, f"{original_id}#orig"
        )
        for v in range(n_variants):
            code = torch.randn(4, generator=generator) * 0.6
            delta = torch.stack([basis[c] @ code for c in (0, 1)])  # [2, H]
            # macro 합이 정확히 delta가 되도록 각 macro 구간의 block에 균등 분배합니다.
            extra = delta.view(1, 1, 2, hidden).expand(
                n_blocks, n_landmarks, 2, hidden
            ) / (n_blocks // n_macro)
            variant = _toy_fine_page(
                n_blocks,
                n_landmarks,
                hidden,
                generator,
                original_id,
                f"{original_id}#var{v}",
                extra_updates=extra,
            )
            # variant 자체의 random update가 relation을 흐리지 않도록 original update를 재사용.
            variant.updates[:] = original.updates + extra
            variant.state[0] = original.state[0]
            for depth in range(n_blocks):
                variant.state[depth + 1] = variant.state[depth] + variant.updates[depth].sum(
                    dim=1
                )
            object.__setattr__(variant, "_fingerprint", None)
            pairs.append((original, variant))
    return {"pairs": pairs, "basis": basis, "n_macro": n_macro}


def make_toy_state_dependent(
    n_originals: int = 6,
    n_variants: int = 2,
    n_blocks: int = 32,
    n_macro: int = DEFAULT_N_MACRO,
    n_landmarks: int = 4,
    hidden: int = 8,
    latent_dim: int = 4,
    seed: int = 0,
) -> dict:
    """Toy B — state-dependent relation.

        ΔU_q = B g(z, original_state_q, original_update_q)

    같은 z가 support macro에서도 관측되므로 cross-macro transport가 가능합니다.
    Toy를 LRT에 유리하게 만들었다는 사실과 real evidence를 혼동하지 않습니다. 이 결과는
    architecture plumbing sanity일 뿐입니다.
    """
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(2, hidden, latent_dim, generator=generator)
    basis = basis / basis.norm(dim=1, keepdim=True)
    gate = torch.randn(latent_dim, hidden, generator=generator) * 0.4
    per_macro = n_blocks // n_macro
    pairs = []
    for index in range(n_originals):
        original_id = f"toyB-{index:03d}"
        original = _toy_fine_page(
            n_blocks, n_landmarks, hidden, generator, original_id, f"{original_id}#orig"
        )
        macro_state = original.state[:: per_macro]  # [G+1, P, H]
        for v in range(n_variants):
            latent = torch.randn(latent_dim, generator=generator) * 0.8
            extra = torch.zeros(n_blocks, n_landmarks, 2, hidden)
            for g in range(n_macro):
                # state 의존 coefficient: latent와 macro 경계 state의 상호작용.
                modulation = torch.tanh(macro_state[g] @ gate.T)  # [P, latent]
                coefficient = latent.view(1, -1) * (1.0 + 0.5 * modulation)  # [P, latent]
                for stream in (0, 1):
                    macro_delta = coefficient @ basis[stream].T  # [P, H]
                    block_share = macro_delta / per_macro
                    extra[g * per_macro : (g + 1) * per_macro, :, stream] = block_share
            variant = _toy_fine_page(
                n_blocks,
                n_landmarks,
                hidden,
                generator,
                original_id,
                f"{original_id}#var{v}",
            )
            variant.updates[:] = original.updates + extra
            variant.state[0] = original.state[0]
            for depth in range(n_blocks):
                variant.state[depth + 1] = variant.state[depth] + variant.updates[depth].sum(
                    dim=1
                )
            object.__setattr__(variant, "_fingerprint", None)
            pairs.append((original, variant))
    return {"pairs": pairs, "basis": basis, "gate": gate, "n_macro": n_macro}


def to_macro_pairs(
    pairs: list[tuple[object, object]], n_macro: int
) -> list[tuple[MacroPage, MacroPage]]:
    return [(to_macro_page(o, n_macro), to_macro_page(v, n_macro)) for o, v in pairs]


def split_pairs_by_original(
    pairs: list[tuple[MacroPage, MacroPage]], holdout: int = 2
) -> tuple[list, list]:
    """original-group split. 같은 original의 variants가 split을 넘지 않습니다."""
    ordered = sorted({o.original_id for o, _ in pairs})
    if len(ordered) <= holdout:
        raise ValueError("not enough originals to hold out a validation split")
    validation_ids = set(ordered[:holdout])
    train = [(o, v) for o, v in pairs if o.original_id not in validation_ids]
    validation = [(o, v) for o, v in pairs if o.original_id in validation_ids]
    return train, validation


# --------------------------------------------------------------------------------------
# macro audit (Fine vs Macro bridge)
# --------------------------------------------------------------------------------------


def audit_macro_page(
    fine_pairs: list[tuple[object, object]],
    n_macro: int = DEFAULT_N_MACRO,
    *,
    rank4: Rank4LinearTransport | None = None,
    stats: MacroNormStats | None = None,
    floor: float | None = None,
) -> dict:
    """MacroPage가 Fine structure를 얼마나 잃는지 보고합니다.

    이 audit은 MacroPage가 U4 signal을 유지한다고 synthetic으로 증명하는 용도가 아니라,
    실제 server Fine Page에서 다음 단계가 평가할 수 있게 만드는 interface입니다.
    """
    from .macro_page import relation_path_energy

    rows = []
    source_blocks = {page.n_blocks for pair in fine_pairs for page in pair}
    too_shallow = sorted(n for n in source_blocks if n < n_macro)
    if too_shallow:
        raise ValueError(
            f"MacroPage needs n_macro <= L but the source has Fine Page depth(s) {too_shallow} "
            f"< n_macro={n_macro}. Pass a deeper Page store, or lower lrt.n_macro / --n-macro "
            "(an even number >= 4)"
        )
    for original, variant in fine_pairs:
        macro_o = to_macro_page(original, n_macro, with_path_energy=True)
        macro_v = to_macro_page(variant, n_macro, with_path_energy=True)
        valid = original.valid & variant.valid
        fine_delta = variant.updates - original.updates
        macro_delta = macro_relation(macro_o, macro_v)
        fine_energy = relation_energy(fine_delta, valid)
        macro_energy = relation_energy(macro_delta, valid)
        boundaries = list(macro_o.boundaries)
        # cancellation diagnostic: macro sum norm 대비 fine path energy.
        pe = relation_path_energy(original, variant, boundaries)
        cancellation = macro_delta.norm(dim=-1) / pe.clamp_min(1e-12)
        rows.append(
            {
                "original_id": original.original_id,
                "variant_id": variant.variant_id,
                "macro_residual_identity_error": max(
                    macro_o.residual_identity_error(), macro_v.residual_identity_error()
                ),
                "boundaries": boundaries,
                "fine_relation_energy": fine_energy,
                "macro_relation_energy": macro_energy,
                "macro_over_fine_energy": (
                    macro_energy / fine_energy if fine_energy > 0 else float("nan")
                ),
                "fine_residual_identity_error": max(original.residual_identity_error(), variant.residual_identity_error()),
                "relation_path_energy_mean": float(pe[:, valid].mean()),
                "cancellation_values": cancellation[:, valid].tolist(),
                "final_token_available": bool(valid[-1]),
                "n_common_valid": int(valid.sum()),
            }
        )
    # identity variant는 fine relation energy가 0이므로 비율이 undefined입니다. 평균에서
    # 제외하고 그 수를 따로 보고합니다.
    ratios = [
        row["macro_over_fine_energy"]
        for row in rows
        if row["macro_over_fine_energy"] == row["macro_over_fine_energy"]
    ]
    summary = {
        "macro_schema_n_macro": n_macro,
        "n_pairs": len(rows),
        "n_pairs_zero_fine_relation_energy": len(rows) - len(ratios),
        "max_macro_residual_identity_error": max(
            (row["macro_residual_identity_error"] for row in rows), default=0.0
        ),
        "mean_macro_over_fine_energy": (
            sum(ratios) / len(ratios) if ratios else float("nan")
        ),
        "pairs": rows,
        "notes": [
            "path_energy는 diagnostic 전용이며 LRT-v1 학습 경로에 들어가지 않습니다.",
            "macro/fine relation energy 비교는 cancellation 정도를 보는 지표입니다.",
            "이 audit은 MacroPage가 signal을 유지한다는 증명이 아닙니다.",
        ],
    }
    if rank4 is not None and stats is not None and floor is not None:
        macro_pairs = to_macro_pairs(fine_pairs, n_macro)
        summary["rank4_macro_transport_ratio"] = evaluate_transport(
            None, macro_pairs, stats, floor=floor, baseline=rank4
        )["transport_ratio"]
    for mode in LANDMARK_MODES:
        counts = []
        for original, variant in fine_pairs:
            macro_o = to_macro_page(original, n_macro)
            macro_v = to_macro_page(variant, n_macro)
            counts.append(int(landmark_mask(macro_o, macro_v, mode).sum()))
        summary[f"{mode}_landmark_count_mean"] = (
            sum(counts) / len(counts) if counts else float("nan")
        )
    return summary


# --------------------------------------------------------------------------------------
# experiment entry point
# --------------------------------------------------------------------------------------


def resolve_floor(lrt_cfg, *, allow_toy_default: bool = False) -> float:
    """denominator floor τ. real data에서 null이면 fail-fast합니다."""
    if lrt_cfg.denominator_floor is None:
        if not allow_toy_default:
            raise ValueError(
                "lrt.denominator_floor is null; a frozen τ must come from a server audit "
                "(repeated FP32 Page numerical noise or a train-only rule) before a real run. "
                "Do not tune it on held-out results"
            )
        raise ValueError("toy fixtures must pass an explicit denominator_floor")
    return float(lrt_cfg.denominator_floor)


def run_toy_experiment(
    lrt_cfg,
    *,
    toy: str = "B",
    seed: int = 0,
    max_epochs: int = 200,
    n_originals: int = 8,
) -> dict:
    """CPU toy에서 LRT와 baseline을 비교합니다. architecture plumbing sanity 전용입니다."""
    floor = resolve_floor(lrt_cfg)
    builder = make_toy_state_dependent if toy.upper() == "B" else make_toy_rank4
    bundle = builder(n_originals=n_originals, n_macro=lrt_cfg.n_macro, seed=seed)
    macro_pairs = to_macro_pairs(bundle["pairs"], lrt_cfg.n_macro)
    train_pairs, validation_pairs = split_pairs_by_original(macro_pairs, holdout=2)
    stats = fit_macro_norm_stats([o for o, _ in train_pairs])

    baselines = {
        "zero": ZeroTransport(),
        "train_mean": TrainMeanTransport.fit(train_pairs, stats),
        "rank4_linear_transport": Rank4LinearTransport.fit(train_pairs, stats),
    }
    results = {
        name: evaluate_transport(
            None,
            validation_pairs,
            stats,
            floor=floor,
            landmark_mode=lrt_cfg.landmark_mode,
            baseline=baseline,
        )
        for name, baseline in baselines.items()
    }
    trained = train_lrt(
        train_pairs,
        validation_pairs,
        stats,
        lrt_cfg,
        floor=floor,
        max_epochs=max_epochs,
        seed=seed,
    )
    results["lrt_v1"] = evaluate_transport(
        trained["model"],
        validation_pairs,
        stats,
        floor=floor,
        landmark_mode=lrt_cfg.landmark_mode,
        support_swap=True,
        eval_seed=seed,
    )
    results["lrt_v1_final_token"] = evaluate_transport(
        trained["model"],
        validation_pairs,
        stats,
        floor=floor,
        landmark_mode="final_token",
        support_swap=True,
        eval_seed=seed,
    )
    return {
        "toy": toy.upper(),
        "schema": LRT_SCHEMA,
        "status": STATUS_CPU_VALIDATED,
        "n_train_pairs": len(train_pairs),
        "n_validation_pairs": len(validation_pairs),
        "stats_hash": stats.hash(),
        "params": trained["params"],
        "epochs_run": trained["epochs_run"],
        "best": trained["best"],
        "results": results,
        "model": trained["model"],
        "stats": stats,
        "notes": [
            "toy 결과는 architecture plumbing sanity이며 real-data evidence가 아닙니다.",
            "cross-macro relational transport이며 future prediction / causal transport가 아닙니다.",
        ],
    }


def command(args) -> int:
    """CLI entry point. 실제 server Page 실험은 실행하지 않습니다."""
    from .config import load_config

    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if getattr(args, "source", None) or getattr(args, "execute_gpu", False):
        from .lrt_real import command as real_command
        return real_command(args)
    cfg = load_config(args.config) if getattr(args, "config", None) else None
    lrt_cfg = cfg.lrt if cfg is not None else None
    if lrt_cfg is None:
        from .config import LRTConfig

        lrt_cfg = LRTConfig()
    if getattr(args, "toy", None):
        # toy fixture는 명시적으로 알려진 τ를 씁니다.
        if lrt_cfg.denominator_floor is None:
            lrt_cfg = replace_floor(lrt_cfg, args.toy_floor)
        report = run_toy_experiment(
            lrt_cfg,
            toy=args.toy,
            seed=getattr(args, "seed", 0),
            max_epochs=getattr(args, "max_epochs", 40),
        )
        model = report.pop("model")
        stats = report.pop("stats")
        save_lrt_checkpoint(
            root / "lrt_toy.pt",
            {"model": model, "best": report["best"], "history": []},
            lrt_cfg,
            stats,
        )
        atomic_write_json(root / "lrt_toy_report.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0
    spec = {
        "schema": LRT_SCHEMA,
        "status": "SPEC_ONLY / SERVER-UNTESTED",
        "lrt_config": {
            "n_macro": lrt_cfg.n_macro,
            "adapter_dim": lrt_cfg.adapter_dim,
            "relation_dim": lrt_cfg.relation_dim,
            "decoder_rank": lrt_cfg.decoder_rank,
            "consistency_weight": lrt_cfg.consistency_weight,
            "support_dropout": lrt_cfg.support_dropout,
            "denominator_floor": lrt_cfg.denominator_floor,
            "landmark_mode": lrt_cfg.landmark_mode,
        },
        "folds": [list(fold) for fold in query_folds(lrt_cfg.n_macro)],
        "server_pending": [
            "frozen denominator floor τ from a server audit",
            "real FP32 Page transport result",
            "Fine vs Macro bridge on real Pages",
        ],
        "note": "실제 Page 실험은 --source를 받는 다음 단계에서 실행합니다.",
    }
    atomic_write_json(root / "lrt_spec.json", spec)
    print(json.dumps(spec, indent=2, ensure_ascii=False))
    return 0


def audit_command(args) -> int:
    """audit-macro-page: Fine vs Macro bridge 보고.

    `--source`가 있으면 그 Page store를 읽고, 없으면 CPU toy fixture로 interface를
    검증합니다. 실제 server Page 평가는 다음 단계입니다.
    """
    from .config import load_config

    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config) if getattr(args, "config", None) else None
    n_macro = getattr(args, "n_macro", None) or (
        cfg.lrt.n_macro if cfg is not None else DEFAULT_N_MACRO
    )
    if getattr(args, "source", None):
        pairs = load_fine_pairs(Path(args.source))
        source = str(args.source)
    else:
        bundle = make_toy_state_dependent(n_originals=4, n_variants=1, n_macro=n_macro, seed=0)
        pairs = bundle["pairs"]
        source = "cpu_toy_fixture"
    try:
        report = audit_macro_page(pairs, n_macro)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    report["source"] = source
    report["status"] = (
        STATUS_CPU_VALIDATED if source == "cpu_toy_fixture" else "REAL_SOURCE_AUDIT"
    )
    atomic_write_json(root / "macro_page_audit.json", report)
    summary = {key: value for key, value in report.items() if key != "pairs"}
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


def load_fine_pairs(source: Path) -> list[tuple[object, object]]:
    """Page store directory에서 (original, variant) Fine Page pair를 읽습니다.

    기존 `load_dataset` 계약을 그대로 씁니다. behavior label 파일은 읽지 않습니다.
    """
    from .data import load_dataset

    index = source / "index.json"
    if not index.exists():
        raise ValueError(
            f"no page store at {source} (missing index.json); pass a directory written by "
            "'aimo extract' / 'aimo make-toy'"
        )
    pairs: list[tuple[object, object]] = []
    for dataset in load_dataset(source).values():
        for group in dataset.groups:
            for variant in group.variants:
                pairs.append((group.original, variant))
    if not pairs:
        raise ValueError(f"page store {source} has no original-variant pair")
    return pairs


def replace_floor(lrt_cfg, floor: float | None):
    """toy 실행용으로 explicit τ를 주입한 복제 config를 만듭니다."""
    from dataclasses import replace

    if floor is None:
        raise ValueError("toy run needs --toy-floor because denominator_floor is null")
    return replace(lrt_cfg, denominator_floor=float(floor))
