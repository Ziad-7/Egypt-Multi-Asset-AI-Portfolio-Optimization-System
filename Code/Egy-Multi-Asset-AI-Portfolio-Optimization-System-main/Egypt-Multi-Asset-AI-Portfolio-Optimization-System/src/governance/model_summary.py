"""Extract a compact regression / classification model summary for outputs/."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.config.settings import MODEL_SUMMARY_PATH, OUTPUTS_DIR

REGRESSION_METRIC_KEYS = (
    "median_r2",
    "median_mae",
    "median_baseline_mae",
    "median_directional_accuracy",
    "composite_score",
    "fold_count",
)

CLASSIFICATION_METRIC_KEYS = (
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "fold_f1_std",
)


def _round_val(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
    if isinstance(value, int):
        return value
    return value


def _summarize_regression(forecasts: Mapping[str, Any]) -> Dict[str, Any]:
    diagnostics = forecasts.get("model_diagnostics") or {}
    best_by_asset = forecasts.get("best_model") or {}
    metrics_r2 = forecasts.get("model_metrics_r2") or {}

    by_asset: Dict[str, Any] = {}
    best_model_wins: Dict[str, int] = {}

    for asset, models in diagnostics.items():
        if not isinstance(models, dict):
            continue
        model_rows: Dict[str, Dict[str, Any]] = {}
        for model_name, stats in models.items():
            if not isinstance(stats, dict):
                continue
            row = {k: _round_val(stats.get(k)) for k in REGRESSION_METRIC_KEYS if k in stats}
            if metrics_r2.get(asset, {}).get(model_name) is not None:
                row["median_r2"] = _round_val(metrics_r2[asset][model_name])
            model_rows[model_name] = row

        best = best_by_asset.get(asset, "none")
        by_asset[asset] = {
            "best_model": best,
            "sample_size": forecasts.get("sample_sizes", {}).get(asset),
            "models": model_rows,
        }
        if isinstance(best, str) and best != "none":
            best_model_wins[best] = best_model_wins.get(best, 0) + 1

    overall_best = None
    if best_model_wins:
        overall_best = max(best_model_wins, key=best_model_wins.get)

    return {
        "horizon_trading_days": 63,
        "by_asset": by_asset,
        "best_model_per_asset": dict(best_by_asset),
        "overall_best_model": overall_best,
        "best_model_win_counts": best_model_wins,
    }


def _summarize_classification(evaluation: Mapping[str, Any], focus_asset: str = "EGX30") -> Dict[str, Any]:
    status = evaluation.get("status", "unknown")
    base: Dict[str, Any] = {
        "focus_asset": focus_asset,
        "status": status,
        "best_model": evaluation.get("best_model"),
        "models": {},
        "ensemble": {},
    }

    if status != "ok":
        base["note"] = f"Classification summary limited: status={status}"
        return base

    models_raw = evaluation.get("model_metrics") or []
    models: Dict[str, Dict[str, Any]] = {}
    if isinstance(models_raw, list):
        for entry in models_raw:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("model", "unknown"))
            models[name] = {
                k: _round_val(entry.get(k))
                for k in CLASSIFICATION_METRIC_KEYS
                if k in entry
            }

    base["models"] = models
    base["ensemble"] = {
        k: _round_val(evaluation.get(k))
        for k in (
            "holdout_size",
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "confidence_calibration_error",
            "fold_f1_std_max",
            "ensemble_disagreement",
        )
        if k in evaluation
    }
    weights = evaluation.get("ensemble_model_weights")
    if isinstance(weights, dict):
        base["ensemble_weights"] = {k: _round_val(v) for k, v in weights.items()}

    return base


def build_model_summary(report: Mapping[str, Any], *, focus_asset: str = "EGX30") -> Dict[str, Any]:
    """Build a standalone summary dict from a full intelligence report payload."""
    meta = report.get("metadata") or {}
    run_meta = meta.get("run_metadata") or {}
    forecasts = (report.get("strategic_diagnostics") or {}).get("forecasts") or {}
    tactical = report.get("tactical_signal") or {}
    evaluation = tactical.get("model_evaluation") or {}

    return {
        "generated_at_utc": run_meta.get("generated_at_utc"),
        "engine_semantic_version": run_meta.get("engine_semantic_version"),
        "universe": meta.get("universe"),
        "strategic_regression": _summarize_regression(forecasts),
        "tactical_classification": _summarize_classification(evaluation, focus_asset=focus_asset),
    }


def write_model_summary(
    report: Mapping[str, Any],
    path: Optional[Path] = None,
    *,
    focus_asset: str = "EGX30",
) -> Path:
    """Write model summary JSON next to the intelligence report."""
    out_path = path or MODEL_SUMMARY_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_model_summary(report, focus_asset=focus_asset)

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        return obj

    out_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return out_path


def default_summary_path_for_report(report_path: Path) -> Path:
    """Place summary beside a custom intelligence report path when provided."""
    if report_path.resolve().parent == OUTPUTS_DIR.resolve():
        return MODEL_SUMMARY_PATH
    return report_path.parent / "model_summary.json"
