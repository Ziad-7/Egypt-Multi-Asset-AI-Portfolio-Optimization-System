"""Tactical signal engine.

Fuses the classifier ensemble, the momentum snapshot, the regime
detector and the latest technical indicators into a single tactical
output.  The engine also computes a *position-size* recommendation
using volatility targeting, so downstream consumers know how much risk
to actually take when the signal fires.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from src.api.schemas import TacticalSignalOutput
from src.tactical.classifiers.models import classify_tactical_signal
from src.tactical.market_regimes.regime_detector import detect_regime_full
from src.tactical.momentum.signals import momentum_snapshot
from src.tactical.technicals.indicators import latest_technical_snapshot


def _strategic_bias_from_weights(weights: Dict[str, float]) -> str:
    egx_weight = weights.get("EGX30", 0.0) + weights.get("EGX100", 0.0)
    safe_weight = weights.get("Gold", 0.0) + weights.get("TBills", 0.0)
    if egx_weight > safe_weight + 0.05:
        return "bullish"
    if safe_weight > egx_weight + 0.05:
        return "defensive"
    return "neutral"


def _in_rebalance_window(next_rebalance_date: str, window_days: int = 14) -> bool:
    today = datetime.now(timezone.utc).date()
    rebalance = datetime.fromisoformat(next_rebalance_date).date()
    return 0 <= (rebalance - today).days <= window_days


def _quality_confidence_scale(quality: Mapping[str, object] | None) -> float:
    """Down-weight tactical confidence when panel QA flags severe issues (metadata-only)."""
    if not quality:
        return 1.0
    scale = 1.0
    for w in quality.get("warnings", []) or []:
        if not isinstance(w, dict):
            continue
        sev = str(w.get("severity", "")).lower()
        if sev == "high":
            scale *= 0.94
        elif sev == "medium":
            scale *= 0.97
    summary = quality.get("asset_summary") or {}
    stale = 0
    for entry in summary.values():
        if not isinstance(entry, dict):
            continue
        qf = entry.get("quality_flags") or {}
        if isinstance(qf, dict) and qf.get("stale_series_flag"):
            stale += 1
    if stale:
        scale *= max(0.72, 1.0 - 0.035 * stale)
    return float(np.clip(scale, 0.62, 1.0))


def _vol_targeted_size(realized_vol_annual: float, target_vol_annual: float = 0.12) -> float:
    """Return a [0, 1.5] sizing multiplier based on volatility targeting."""
    if realized_vol_annual <= 1e-6:
        return 1.0
    raw = target_vol_annual / realized_vol_annual
    return float(np.clip(raw, 0.20, 1.50))


def build_tactical_signal(
    features: Dict[str, pd.DataFrame],
    focus_asset: str = "EGX30",
    strategic_weights: Dict[str, float] | None = None,
    next_rebalance_date: str | None = None,
    target_vol_annual: float = 0.18,
    panel_quality: Mapping[str, object] | None = None,
) -> TacticalSignalOutput:
    df = features[focus_asset]
    cls = classify_tactical_signal(df)
    momentum = momentum_snapshot(df)
    regime = detect_regime_full(df)
    technicals = latest_technical_snapshot(df)

    blended = round(0.6 * cls["signal"] + 0.4 * momentum.signal)
    signal = int(max(-1, min(1, blended)))
    confidence = float(cls["confidence"])
    rebalance_window_active = False

    if regime.risk_off:
        signal = -1
        confidence = max(confidence, 0.7)
    elif regime.regime == "bear_trend" and signal > 0:
        # Don't fight a bear trend with a long tactical signal.
        signal = 0
        confidence *= 0.85

    if strategic_weights and next_rebalance_date:
        bias = _strategic_bias_from_weights(strategic_weights)
        rebalance_window_active = _in_rebalance_window(next_rebalance_date)
        if rebalance_window_active:
            aligned = (
                (bias == "bullish" and signal >= 0)
                or (bias == "defensive" and signal <= 0)
                or bias == "neutral"
            )
            confidence = float(np.clip(confidence * (1.10 if aligned else 0.85), 0.05, 0.98))
            if not aligned and signal == 1 and bias == "defensive":
                signal = 0

    suggested_size = _vol_targeted_size(regime.realized_vol, target_vol_annual=target_vol_annual)
    if signal == -1:
        suggested_size *= 0.5  # de-risk on sell signal
    if regime.risk_off:
        suggested_size = min(suggested_size, 0.3)

    qscale = _quality_confidence_scale(panel_quality)
    confidence = float(np.clip(confidence * qscale, 0.05, 0.98))

    rationale = [
        f"Regime={regime.regime}",
        f"RealizedVolAnn={regime.realized_vol:.3f}",
        f"LongVolAnn={regime.long_vol:.3f}",
        f"Drawdown={regime.drawdown:.3f}",
        f"Classifier={cls['signal']}",
        f"MomentumSignal={momentum.signal}",
        f"MomentumStrength={momentum.strength:.3f}",
        f"RSI={technicals.get('rsi', 50):.2f}",
        f"RebalanceWindow={rebalance_window_active}",
        f"PanelQualityScale={qscale:.3f}",
    ]
    return TacticalSignalOutput(
        signal=signal,
        confidence=round(confidence, 4),
        regime=regime.regime,
        risk_off_alert=regime.risk_off,
        rebalance_window_active=rebalance_window_active,
        rationale=rationale,
        suggested_position_size=round(float(suggested_size), 4),
        target_volatility=round(float(target_vol_annual), 4),
        realized_volatility=round(float(regime.realized_vol), 4),
        model_evaluation=cls.get("evaluation", {}),
    )
