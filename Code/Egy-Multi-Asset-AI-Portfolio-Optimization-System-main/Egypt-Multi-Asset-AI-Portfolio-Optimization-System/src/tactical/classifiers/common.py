from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelEvalResult:
    model_name: str
    latest_probability: np.ndarray | None
    oof_probabilities: np.ndarray | None
    metric: dict[str, float]
    fold_metrics: list[dict[str, float | int | str]]
    weight: float


def clip_features(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.nanquantile(train, 0.01, axis=0)
    hi = np.nanquantile(train, 0.99, axis=0)
    return np.clip(train, lo, hi), np.clip(test, lo, hi)


def model_weight_from_metrics(f1_weighted: float, fold_f1_std: float) -> float:
    stability_penalty = float(np.clip(1.0 - 2.0 * fold_f1_std, 0.45, 1.0))
    return float(max(0.05, f1_weighted * stability_penalty))


def weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return float(precision_w), float(recall_w), float(f1_w)


def scale_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_train, X_test = clip_features(X_train, X_test)
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)
