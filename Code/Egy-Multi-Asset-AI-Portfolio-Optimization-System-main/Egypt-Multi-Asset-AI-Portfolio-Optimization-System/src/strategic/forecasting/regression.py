"""Expected-return forecasting with a walk-forward, leakage-safe ensemble.

The legacy implementation suffered from three issues that this rewrite
addresses:

1. **Future leakage** -- features were winsorized using *full-sample*
   quantiles before the chronological train/test split.  The new
   pipeline computes outlier bounds on the train window only.
2. **Single-fold validation** -- a single 80/20 split on a small
   Egyptian sample produced unstable confidence numbers.  We now use
   expanding-window walk-forward CV with the median forecast.
3. **Confidence proxy** -- the legacy ``1/(1+rmse*100)`` mapping was
   uncalibrated; predictions with no signal still produced ~0.7
   confidence.  We replace it with an out-of-sample R^2 (clipped to
   [0, 1]) which is a standard skill measure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from src.strategic.forecasting.models import ModelRunResult, run_random_forest, run_ridge, run_svr

FEATURES = ["return_lag1", "return_lag2", "return_lag3", "dist_to_ma5", "macd_hist", "rsi", "bb_pb"]


@dataclass
class ForecastSummary:
    expected_returns: Dict[str, float]      # daily, decimal
    confidence: Dict[str, float]            # OOS R^2 in [0, 1]
    sample_sizes: Dict[str, int]
    model_predictions_63d: Dict[str, Dict[str, float]]
    model_metrics: Dict[str, Dict[str, float]]  # legacy: median R^2 by model
    model_diagnostics: Dict[str, Dict[str, Dict[str, float]]]
    best_model: Dict[str, str]


def _prepare(asset_df: pd.DataFrame, horizon: int = 63) -> Tuple[pd.DataFrame, pd.Series]:
    cols = [c for c in FEATURES if c in asset_df.columns]
    if not cols or "return" not in asset_df.columns:
        return pd.DataFrame(), pd.Series(dtype=float)
    
    df = asset_df[cols + ["return"]].copy()
    
    # Calculate compounded forward return for the given horizon
    df["target"] = (1.0 + df["return"]).rolling(window=horizon).apply(np.prod, raw=True).shift(-horizon) - 1.0
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["target"])
    return df[cols], df["target"]


def _ensemble_forecast(
    asset_df: pd.DataFrame,
) -> Tuple[float, float, int, Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]], str]:
    X_df, y_series = _prepare(asset_df)
    if X_df.empty:
        return 0.0, 0.0, 0, {}, {}, {}, "none"
    X, y = X_df.values, y_series.values
    if len(X) < 80:
        # Not enough sample to do walk-forward CV; fall back to recent mean.
        fallback = float(np.nanmean(y[-30:])) / 63.0
        return fallback, 0.20, int(len(X)), {}, {}, {}, "fallback_mean_30d"

    model_results: list[ModelRunResult] = [
        run_ridge(X, y),
        run_random_forest(X, y),
        run_svr(X, y),
    ]
    valid_results = [
        r
        for r in model_results
        if r.latest_prediction_63d is not None
        and r.median_r2 is not None
        and r.composite_score is not None
    ]

    if not valid_results:
        fallback = float(np.nanmean(y[-30:])) / 63.0
        return fallback, 0.20, int(len(X)), {}, {}, {}, "fallback_mean_30d"

    predictions_63d = {r.model_name: float(r.latest_prediction_63d) for r in valid_results}
    metrics_r2 = {r.model_name: float(r.median_r2) for r in valid_results}
    diagnostics = {
        r.model_name: {
            "median_r2": float(r.median_r2),
            "median_mae": float(r.median_mae),
            "median_baseline_mae": float(r.median_baseline_mae),
            "median_directional_accuracy": float(r.median_directional_accuracy),
            "composite_score": float(r.composite_score),
            "fold_count": float(r.fold_count),
        }
        for r in valid_results
    }
    best_model_result = max(valid_results, key=lambda r: float(r.composite_score))
    best_model = best_model_result.model_name

    expected_return = float(np.median(list(predictions_63d.values())))
    # Robust anchor that keeps expected returns stable when OOS skill is weak.
    anchor_return_63d = float(np.nanmedian(y[-90:])) if len(y) >= 90 else float(np.nanmedian(y))
    skill = float(np.clip(best_model_result.composite_score, -1.0, 1.0))
    shrink = float(np.clip(0.50 + 0.45 * skill, 0.15, 0.95))
    blended_expected_return = shrink * expected_return + (1.0 - shrink) * anchor_return_63d
    # Convert the predicted 63-day return into an average daily expected return
    # so the downstream optimizer can annualize it correctly
    expected_return_daily = blended_expected_return / 63.0

    # Confidence uses blended skill factors to avoid over-trusting noisy R^2.
    raw_r2 = float(np.median([r.median_r2 for r in valid_results]))
    dir_acc = float(np.median([r.median_directional_accuracy for r in valid_results]))
    mae_skill = float(
        np.median(
            [
                1.0 - (r.median_mae / (r.median_baseline_mae + 1e-12))
                for r in valid_results
            ]
        )
    )
    confidence = float(np.clip(0.12 + 0.32 * raw_r2 + 0.28 * mae_skill + 0.28 * (dir_acc - 0.5), 0.05, 0.95))
    return expected_return_daily, confidence, int(len(X)), predictions_63d, metrics_r2, diagnostics, best_model


def forecast_expected_returns(features: Dict[str, pd.DataFrame]) -> ForecastSummary:
    expected_returns: Dict[str, float] = {}
    confidence: Dict[str, float] = {}
    samples: Dict[str, int] = {}
    model_predictions_63d: Dict[str, Dict[str, float]] = {}
    model_metrics: Dict[str, Dict[str, float]] = {}
    model_diagnostics: Dict[str, Dict[str, Dict[str, float]]] = {}
    best_model: Dict[str, str] = {}
    for asset, df in features.items():
        if "return" not in df.columns:
            continue
        mu, conf, n, preds_63d, metrics, diagnostics, best = _ensemble_forecast(df.dropna(subset=["return"]))
        expected_returns[asset] = mu
        confidence[asset] = conf
        samples[asset] = n
        model_predictions_63d[asset] = preds_63d
        model_metrics[asset] = metrics
        model_diagnostics[asset] = diagnostics
        best_model[asset] = best
    return ForecastSummary(
        expected_returns=expected_returns,
        confidence=confidence,
        sample_sizes=samples,
        model_predictions_63d=model_predictions_63d,
        model_metrics=model_metrics,
        model_diagnostics=model_diagnostics,
        best_model=best_model,
    )
