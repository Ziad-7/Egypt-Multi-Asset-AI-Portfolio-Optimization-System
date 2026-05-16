from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class MacroSignal:
    trend_score: Dict[str, float]
    inflation_pressure: float
    rates_pressure: float


def build_macro_signal(features: Dict[str, pd.DataFrame]) -> MacroSignal:
    trend_score: Dict[str, float] = {}
    inflation_proxy = []
    rates_proxy = []

    for asset, df in features.items():
        if "return" not in df.columns:
            continue
        series = df["return"].dropna().tail(63)
        trend_score[asset] = float(np.tanh(series.mean() / (series.std() + 1e-9)))
        if "gold" in asset.lower():
            inflation_proxy.append(series.mean())
        if "tbill" in asset.lower():
            rates_proxy.append(series.mean())

    inflation_pressure = float(np.mean(inflation_proxy) if inflation_proxy else 0.0)
    rates_pressure = float(np.mean(rates_proxy) if rates_proxy else 0.0)
    return MacroSignal(
        trend_score=trend_score,
        inflation_pressure=inflation_pressure,
        rates_pressure=rates_pressure,
    )
