"""Momentum signal generator with symmetric thresholds and z-score scaling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MomentumSnapshot:
    signal: int
    strength: float
    short_mean: float
    medium_mean: float
    long_mean: float


def momentum_snapshot(asset_df: pd.DataFrame) -> MomentumSnapshot:
    """Compute a 3-horizon momentum signal.

    Compares 5d / 21d / 63d realized means.  The signal is +1 when both
    the short and medium horizons are positive *and* the short horizon
    leads, -1 in the symmetric bearish case, and 0 otherwise.  Strength
    is the standardized short-horizon excess over the medium horizon and
    is exposed for downstream sizing.
    """
    rets = asset_df["return"].dropna()
    if len(rets) < 30:
        return MomentumSnapshot(signal=0, strength=0.0, short_mean=0.0, medium_mean=0.0, long_mean=0.0)

    short = float(rets.tail(5).mean())
    medium = float(rets.tail(21).mean())
    long_ = float(rets.tail(63).mean()) if len(rets) >= 63 else medium
    sigma = float(rets.tail(63).std()) if len(rets) >= 63 else float(rets.std())

    excess = short - medium
    strength = excess / (sigma + 1e-9)

    if short > 0 and medium > 0 and excess > 0:
        signal = 1
    elif short < 0 and medium < 0 and excess < 0:
        signal = -1
    else:
        signal = 0
    return MomentumSnapshot(signal=signal, strength=float(strength), short_mean=short, medium_mean=medium, long_mean=long_)


def momentum_signal(asset_df: pd.DataFrame) -> int:
    """Backward-compatible scalar shim; prefer :func:`momentum_snapshot`."""
    return momentum_snapshot(asset_df).signal
