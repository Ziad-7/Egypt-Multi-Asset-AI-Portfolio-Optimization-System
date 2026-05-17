from __future__ import annotations

from src.governance.model_summary import build_model_summary


def test_build_model_summary_extracts_regression_and_classification():
    report = {
        "metadata": {
            "universe": ["EGX30", "Gold"],
            "run_metadata": {"generated_at_utc": "2026-01-01T00:00:00+00:00"},
        },
        "strategic_diagnostics": {
            "forecasts": {
                "sample_sizes": {"EGX30": 100},
                "best_model": {"EGX30": "ridge"},
                "model_metrics_r2": {"EGX30": {"ridge": -0.1, "random_forest": -0.2}},
                "model_diagnostics": {
                    "EGX30": {
                        "ridge": {
                            "median_r2": -0.1,
                            "median_mae": 0.11,
                            "median_baseline_mae": 0.12,
                            "median_directional_accuracy": 0.51,
                            "composite_score": -0.2,
                            "fold_count": 4.0,
                        },
                        "random_forest": {
                            "median_r2": -0.2,
                            "median_mae": 0.12,
                            "median_baseline_mae": 0.12,
                            "median_directional_accuracy": 0.50,
                            "composite_score": -0.25,
                            "fold_count": 4.0,
                        },
                    }
                },
            }
        },
        "tactical_signal": {
            "model_evaluation": {
                "status": "ok",
                "best_model": "logistic_regression",
                "accuracy": 0.35,
                "f1_weighted": 0.34,
                "model_metrics": [
                    {
                        "model": "logistic_regression",
                        "accuracy": 0.35,
                        "f1_weighted": 0.34,
                        "fold_f1_std": 0.03,
                    },
                    {
                        "model": "xgboost",
                        "accuracy": 0.33,
                        "f1_weighted": 0.32,
                        "fold_f1_std": 0.04,
                    },
                ],
                "ensemble_model_weights": {"logistic_regression": 0.6, "xgboost": 0.4},
            }
        },
    }
    summary = build_model_summary(report)
    assert summary["strategic_regression"]["best_model_per_asset"]["EGX30"] == "ridge"
    assert summary["strategic_regression"]["overall_best_model"] == "ridge"
    assert "ridge" in summary["strategic_regression"]["by_asset"]["EGX30"]["models"]
    assert summary["tactical_classification"]["best_model"] == "logistic_regression"
    assert summary["tactical_classification"]["models"]["xgboost"]["f1_weighted"] == 0.32
