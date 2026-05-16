"""Market regime detector using volatility, trend and drawdown features.

Identifies four regimes:

* ``bull_trend`` -- realized vol below long-vol benchmark, trend > 0
* ``high_volatility_risk_off`` -- realized vol >> long-vol AND trend negative
* ``bear_trend`` -- trend < 0 with elevated vol
* ``sideways`` -- low vol, low trend

The detector is intentionally rule-based for transparency.  An HMM-style
two-state classifier could improve smoothness but would require more
sample than the Egyptian universe currently provides.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeAssessment:
    regime: str
    risk_off: bool
    realized_vol: float          # annualized, decimal
    long_vol: float              # annualized, decimal
    trend: float                 # average daily return last 20d
    drawdown: float              # latest drawdown vs trailing-1y peak


def detect_regime_full(asset_df: pd.DataFrame) -> RegimeAssessment:
    returns = asset_df["return"].dropna()
    if len(returns) < 30:
        return RegimeAssessment("uncertain", False, 0.0, 0.0, 0.0, 0.0)

    realized_vol = float(returns.tail(20).std() * np.sqrt(252))
    long_vol = float(returns.tail(120).std() * np.sqrt(252)) if len(returns) >= 120 else float(returns.std() * np.sqrt(252))
    trend = float(returns.tail(20).mean())

    cum = (1.0 + returns).cumprod()
    lookback = min(252, len(cum))
    peak = cum.iloc[-lookback:].max()
    drawdown = float(cum.iloc[-1] / peak - 1.0)

    risk_off = realized_vol > 1.30 * (long_vol + 1e-9) and (trend < 0 or drawdown < -0.08)
    if risk_off:
        regime = "high_volatility_risk_off"
    elif trend > 0 and realized_vol < long_vol * 1.10:
        regime = "bull_trend"
    elif trend < 0:
        regime = "bear_trend"
    else:
        regime = "sideways"
    return RegimeAssessment(regime=regime, risk_off=risk_off, realized_vol=realized_vol, long_vol=long_vol, trend=trend, drawdown=drawdown)


def detect_regime(asset_df: pd.DataFrame) -> tuple[str, bool]:
    """Backward-compatible 2-tuple shim."""
    a = detect_regime_full(asset_df)
    return a.regime, a.risk_off
