from __future__ import annotations

from src.governance.metadata import ENGINE_SEMANTIC_VERSION, collect_run_metadata
from src.tactical.classifiers.models import classify_tactical_signal
from src.tactical.signals.engine import _quality_confidence_scale, build_tactical_signal


def test_collect_run_metadata_required_keys():
    meta = collect_run_metadata()
    assert "generated_at_utc" in meta
    assert meta.get("random_seed_used") is not None
    assert meta.get("engine_semantic_version") == ENGINE_SEMANTIC_VERSION


def test_quality_scale_reduces_with_high_severity_warnings():
    q = {
        "warnings": [{"severity": "high", "code": "x", "message": "y"}],
        "asset_summary": {},
    }
    assert _quality_confidence_scale(q) < 1.0


def test_classify_tactical_signal_returns_evaluation(market_panel):
    out = classify_tactical_signal(market_panel.features["EGX30"])
    assert "evaluation" in out
    assert "status" in out["evaluation"]


def test_build_tactical_signal_accepts_panel_quality(market_panel):
    sig = build_tactical_signal(
        market_panel.features,
        focus_asset="EGX30",
        panel_quality=market_panel.quality.to_dict(),
    )
    assert sig.signal in (-1, 0, 1)
    assert 0.05 <= sig.confidence <= 0.98
