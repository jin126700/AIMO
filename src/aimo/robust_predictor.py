"""작은 regularized logistic robustness predictor.

공간(Stage-E encoder, rank)과 feature 정의를 freeze한 뒤에만 학습합니다. scaling,
정규화 강도, threshold는 **root-grouped validation**에서만 고릅니다. official score나
held-out 결과를 보고 encoder / rank / feature를 다시 맞추지 않습니다.

artifact는 JSON입니다. 공식 runtime과 로컬의 scikit-learn 버전이 달라도 되도록 pickle을
쓰지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

FEATURES_CORE = ("prompt_end_divergence", "residual_max", "residual_rms")
FEATURES_CONTROL = (
    "n_tokens",
    "prompt_end_entropy",
    "prompt_end_max_prob",
    "alignment_coverage",
    "n_views",
)
FEATURE_SETS = {
    "core3": FEATURES_CORE,
    "core3_controls": FEATURES_CORE + FEATURES_CONTROL,
    # baseline과 ablation
    "divergence_only": ("prompt_end_divergence",),
    "length_uncertainty": ("n_tokens", "prompt_end_entropy", "prompt_end_max_prob"),
    "prompt_end_only": ("prompt_end_divergence", "prompt_end_entropy", "prompt_end_max_prob"),
    "all_token_only": ("residual_max", "residual_rms"),
}
DEFAULT_FEATURE_SET = "core3"
FIT_SPLITS = ("train", "dev", "fit")


@dataclass
class LogisticPredictor:
    feature_names: list[str]
    transform: str  # "none" | "log1p"
    center: list[float]
    scale: list[float]
    weights: list[float]
    bias: float
    threshold: float
    l2: float

    def _matrix(self, rows: list[dict]) -> np.ndarray:
        x = np.array([[float(row[name]) for name in self.feature_names] for row in rows],
                     dtype=np.float64).reshape(len(rows), len(self.feature_names))
        if self.transform == "log1p":
            x = np.sign(x) * np.log1p(np.abs(x))
        return (x - np.array(self.center)) / np.array(self.scale)

    def predict_proba(self, rows: list[dict]) -> np.ndarray:
        logits = self._matrix(rows) @ np.array(self.weights) + self.bias
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, rows: list[dict]) -> list[bool]:
        return [bool(p >= self.threshold) for p in self.predict_proba(rows)]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> LogisticPredictor:
        return cls(**payload)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _transform(x: np.ndarray, transform: str) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x)) if transform == "log1p" else x


def fit_logistic(
    x: np.ndarray, y: np.ndarray, l2: float, iters: int = 50
) -> tuple[np.ndarray, float]:
    """L2 logistic regression (Newton / IRLS). bias는 정규화하지 않습니다."""
    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xb @ w)))
        grad = xb.T @ (p - y) / n + reg * w
        hess = (xb * (p * (1 - p))[:, None]).T @ xb / n + np.diag(reg) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w[:-1], float(w[-1])


def grouped_folds(groups: list[str], k: int, seed: int = 0) -> list[np.ndarray]:
    """root 단위 fold. 같은 root의 모든 행(model / effort 포함)은 같은 fold입니다."""
    unique = sorted(set(groups), key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    k = max(2, min(k, len(unique)))
    fold_of = {g: i % k for i, g in enumerate(unique)}
    labels = np.array([fold_of[g] for g in groups])
    return [np.where(labels == i)[0] for i in range(k)]


def _best_threshold(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate([prob, [0.5]]))
    best = (0.5, -1.0)
    for t in candidates:
        acc = float(((prob >= t) == y.astype(bool)).mean())
        if acc > best[1] + 1e-12:
            best = (float(t), acc)
    return best


def select_predictor(
    rows: list[dict],
    labels: list[bool],
    groups: list[str],
    *,
    feature_set: str = DEFAULT_FEATURE_SET,
    l2_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
    transforms: tuple[str, ...] = ("none", "log1p"),
    k: int = 5,
    seed: int = 0,
) -> tuple[LogisticPredictor, dict]:
    """root-grouped CV로 (transform, l2)와 threshold를 고르고 전체 fit row로 다시 학습합니다."""
    bad = sorted({row.get("split") for row in rows} - set(FIT_SPLITS) - {None})
    if bad:
        raise ValueError(f"predictor selection must not see split(s) {bad}")
    if len(rows) != len(labels) or len(rows) != len(groups):
        raise ValueError("rows, labels and groups must have the same length")
    names = list(FEATURE_SETS[feature_set])
    raw = np.array([[float(row[n]) for n in names] for row in rows], dtype=np.float64)
    y = np.array([1.0 if label else 0.0 for label in labels])
    folds = grouped_folds(groups, k, seed)
    trials = []
    for transform in transforms:
        x_all = _transform(raw, transform)
        for l2 in l2_grid:
            oof = np.zeros(len(rows))
            for held in folds:
                fit = np.setdiff1d(np.arange(len(rows)), held)
                mu, sd = x_all[fit].mean(0), x_all[fit].std(0) + 1e-9
                w, b = fit_logistic((x_all[fit] - mu) / sd, y[fit], l2)
                oof[held] = 1.0 / (1.0 + np.exp(-(((x_all[held] - mu) / sd) @ w + b)))
            threshold, acc = _best_threshold(oof, y)
            log_loss = float(-np.mean(y * np.log(oof + 1e-12) + (1 - y) * np.log(1 - oof + 1e-12)))
            trials.append({"transform": transform, "l2": l2, "threshold": threshold,
                           "grouped_cv_accuracy": acc, "grouped_cv_log_loss": log_loss})
    best = min(trials, key=lambda t: (-t["grouped_cv_accuracy"], t["grouped_cv_log_loss"]))
    x_all = _transform(raw, best["transform"])
    mu, sd = x_all.mean(0), x_all.std(0) + 1e-9
    w, b = fit_logistic((x_all - mu) / sd, y, best["l2"])
    predictor = LogisticPredictor(
        feature_names=names, transform=best["transform"], center=mu.tolist(), scale=sd.tolist(),
        weights=w.tolist(), bias=b, threshold=best["threshold"], l2=best["l2"],
    )
    report = {
        "feature_set": feature_set,
        "n_rows": len(rows),
        "n_groups": len(set(groups)),
        "n_folds": len(folds),
        "positive_rate": float(y.mean()) if len(y) else math.nan,
        "trials": trials,
        "selected": best,
        "note": "selected on root-grouped validation only; not re-tuned on official scores",
    }
    return predictor, report


def fit_prior(rows: list[dict], labels: list[bool]) -> dict:
    """fallback prior. (model, effort) -> model -> global 순서의 다수결과 그 train 정확도."""
    def majority(values: list[bool]) -> tuple[bool, float]:
        pos = sum(values)
        choice = pos * 2 >= len(values)
        return bool(choice), (pos if choice else len(values) - pos) / max(len(values), 1)

    by_key: dict[str, list[bool]] = {}
    for row, label in zip(rows, labels, strict=True):
        for key in (f"{row['model_id']}|{row['reasoning_effort']}", row["model_id"], "*"):
            by_key.setdefault(key, []).append(bool(label))
    prior = {}
    for key, values in by_key.items():
        choice, acc = majority(values)
        prior[key] = {"prediction": choice, "support": len(values), "train_accuracy": acc}
    return prior
