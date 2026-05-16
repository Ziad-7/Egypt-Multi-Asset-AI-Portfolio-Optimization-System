from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelRunResult:
    model_name: str
    latest_prediction_63d: float | None
    median_r2: float | None
    median_mae: float | None
    median_baseline_mae: float | None
    median_directional_accuracy: float | None
    composite_score: float | None
    fold_count: int


def winsorize_train(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.quantile(X_train, 0.01, axis=0)
    hi = np.quantile(X_train, 0.99, axis=0)
    return np.clip(X_train, lo, hi), np.clip(X_test, lo, hi)


def walk_forward_model_eval(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model_factory: Callable[[], object],
    n_folds: int = 4,
    min_train: int = 60,
) -> ModelRunResult:
    n = len(X)
    if n < min_train + 30:
        return ModelRunResult(
            model_name=model_name,
            latest_prediction_63d=None,
            median_r2=None,
            median_mae=None,
            median_baseline_mae=None,
            median_directional_accuracy=None,
            composite_score=None,
            fold_count=0,
        )

    fold_size = max(15, (n - min_train) // n_folds)
    r2_scores: list[float] = []
    mae_scores: list[float] = []
    baseline_mae_scores: list[float] = []
    directional_scores: list[float] = []
    latest_pred: float | None = None

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if train_end >= n or test_end <= train_end:
            break

        X_tr, X_te = X[:train_end], X[train_end:test_end]
        y_tr, y_te = y[:train_end], y[train_end:test_end]

        X_tr, X_te = winsorize_train(X_tr, X_te)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        try:
            model = model_factory()
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_te_s)
            if not np.all(np.isfinite(preds)):
                continue
            ss_res = float(np.sum((y_te - preds) ** 2))
            ss_tot = float(np.sum((y_te - y_te.mean()) ** 2)) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            r2_scores.append(float(np.clip(r2, -1.0, 1.0)))

            mae = float(np.mean(np.abs(y_te - preds)))
            baseline_pred = float(np.mean(y_tr))
            baseline_mae = float(np.mean(np.abs(y_te - baseline_pred)))
            mae_scores.append(mae)
            baseline_mae_scores.append(baseline_mae)
            directional_scores.append(float(np.mean(np.sign(preds) == np.sign(y_te))))

            if fold == n_folds - 1 or test_end == n:
                latest_pred = float(preds[-1])
        except Exception:
            continue

    if not r2_scores:
        return ModelRunResult(
            model_name=model_name,
            latest_prediction_63d=None,
            median_r2=None,
            median_mae=None,
            median_baseline_mae=None,
            median_directional_accuracy=None,
            composite_score=None,
            fold_count=0,
        )

    med_r2 = float(np.median(r2_scores))
    med_mae = float(np.median(mae_scores))
    med_baseline_mae = float(np.median(baseline_mae_scores))
    med_directional = float(np.median(directional_scores))

    # Composite skill balances baseline-relative error, directional hit-rate,
    # and R^2 so model choice remains robust for noisy financial series.
    mae_skill = 1.0 - (med_mae / (med_baseline_mae + 1e-12))
    dir_skill = 2.0 * (med_directional - 0.5)
    composite = 0.45 * med_r2 + 0.35 * mae_skill + 0.20 * dir_skill

    return ModelRunResult(
        model_name=model_name,
        latest_prediction_63d=latest_pred,
        median_r2=med_r2 if latest_pred is not None else None,
        median_mae=med_mae if latest_pred is not None else None,
        median_baseline_mae=med_baseline_mae if latest_pred is not None else None,
        median_directional_accuracy=med_directional if latest_pred is not None else None,
        composite_score=float(composite) if latest_pred is not None else None,
        fold_count=len(r2_scores),
    )
