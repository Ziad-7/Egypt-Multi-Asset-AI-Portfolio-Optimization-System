"""Macro-data feeds for the Egyptian platform.

This module replaces the brittle "piecewise-constant T-Bill schedule
hard-coded in settings" pattern with a feed abstraction so the platform
can transparently swap between

* a static schedule (good for offline / legacy reproducibility),
* a stochastic OU process anchored to the schedule (used by default to
  capture intra-year auction-cycle volatility),
* and -- in production -- a live CBE primary-auction connector.

It also synthesizes the two macro series the data set lacks:

* ``USDEGPRateFeed`` -- a CIB-GDR-style USD/EGP parallel rate built from
  documented devaluation regime breaks plus an AR(1) parallel-market
  premium and slow log-Brownian drift.  Used as both an optional
  tradable hedge sleeve and as the deflator that turns nominal EGP
  returns into USD-real returns.
* ``EgyptCPIFeed`` -- a piecewise YoY inflation series calibrated to
  CAPMAS / IMF observations, exposing daily CPI level, YoY inflation,
  real yield (T-Bill carry minus inflation), and a regime classifier
  the tactical layer uses to bias the suggested tilt.

All randomness flows through a single ``np.random.default_rng`` seeded
by ``RANDOM_SEED`` from settings so reports are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Protocol, Tuple

import numpy as np
import pandas as pd

from src.config.settings import (
    EGYPT_TBILL_YIELD_SCHEDULE,
    RANDOM_SEED,
    TRADING_DAYS,
    get_tbill_yield,
)


# ---------------------------------------------------------------------------
# T-Bill yield feeds
# ---------------------------------------------------------------------------


class TBillYieldFeed(Protocol):
    """Read-side interface for any T-Bill yield source."""

    def daily_series(self, index: pd.DatetimeIndex) -> pd.Series:
        ...


@dataclass
class StaticScheduleFeed:
    """Piecewise-constant CBE yield schedule (legacy behavior)."""

    def daily_series(self, index: pd.DatetimeIndex) -> pd.Series:
        values = [get_tbill_yield(int(idx.year)) for idx in index]
        return pd.Series(values, index=index, name="tbill_yield")


@dataclass
class OUSimulatedFeed:
    """Mean-reverting OU yield process anchored to the CBE schedule.

    The schedule supplies the long-run mean ``mu(t)``, which steps up or
    down at year boundaries.  Around that mean, yields drift via an
    Ornstein-Uhlenbeck process with mean-reversion strength ``kappa``
    (half-life ~5 weeks, matching CBE auction cadence) and instantaneous
    vol ``sigma`` calibrated so the realised annualised yield vol comes
    out at ~150 bps -- consistent with empirical CBE auction-cycle
    fluctuation.  Floored at 4 % and capped at 35 % to keep us in
    economically defensible territory.
    """

    kappa: float = 8.0      # mean-reversion speed (per year)
    sigma: float = 0.018    # instantaneous vol (per sqrt-year)
    floor: float = 0.04
    cap: float = 0.35
    seed: int = RANDOM_SEED

    def daily_series(self, index: pd.DatetimeIndex) -> pd.Series:
        if len(index) == 0:
            return pd.Series(dtype=float, name="tbill_yield")
        rng = np.random.default_rng(self.seed)
        anchors = np.array([get_tbill_yield(int(idx.year)) for idx in index])
        n = len(index)
        dt = 1.0 / TRADING_DAYS
        y = np.empty(n)
        # Initialize at the first anchor.
        y[0] = anchors[0]
        diffusion = self.sigma * np.sqrt(dt)
        drift = self.kappa * dt
        for t in range(1, n):
            shock = rng.standard_normal()
            y[t] = y[t - 1] + drift * (anchors[t] - y[t - 1]) + diffusion * shock
            y[t] = float(np.clip(y[t], self.floor, self.cap))
        return pd.Series(y, index=index, name="tbill_yield")


def _build_tbill_feed(name: str) -> TBillYieldFeed:
    if name == "static":
        return StaticScheduleFeed()
    if name == "ou_simulated":
        return OUSimulatedFeed()
    raise ValueError(f"Unknown TBILL_FEED setting: {name!r}")


def yield_to_daily_hpr(yield_series: pd.Series) -> pd.Series:
    """Convert annualized yields to a rolling-T-Bill daily HPR series."""
    daily = (1.0 + yield_series.astype(float)) ** (1.0 / TRADING_DAYS) - 1.0
    daily.name = "return"
    return daily


# ---------------------------------------------------------------------------
# USD/EGP feed
# ---------------------------------------------------------------------------


# Documented EGP regime-anchor levels.  The platform synthesizes a daily
# series anchored to these breakpoints; in production they would be
# replaced by CIB GDR-implied rates or NDF parallel-market quotes.
USDEGP_REGIME_ANCHORS: List[Tuple[str, float]] = [
    ("2015-01-01",  7.15),
    ("2016-03-15",  8.85),
    ("2016-11-03", 17.50),   # CBE float
    ("2017-01-01", 18.50),
    ("2018-01-01", 17.80),
    ("2020-01-01", 15.95),
    ("2022-03-21", 18.40),
    ("2022-10-27", 24.40),
    ("2023-01-04", 30.50),
    ("2024-03-06", 49.50),   # IMF-aligned float
    ("2025-01-01", 49.80),
    ("2026-04-30", 50.20),
]


@dataclass
class USDEGPRateFeed:
    """Synthetic CIB-GDR-style USD/EGP parallel rate generator.

    Between regime anchors the level follows a slow log-Brownian drift
    (annualized vol ~6 %) plus an AR(1) parallel-market premium with
    half-life ~30 days.  At each anchor date the level snaps to the
    documented breakpoint (modeling the actual CBE devaluation events).
    """

    annual_drift_vol: float = 0.06
    premium_sigma: float = 0.04
    premium_phi: float = np.exp(-np.log(2.0) / 30.0)  # 30-day half-life
    seed: int = RANDOM_SEED + 1

    def daily_series(self, index: pd.DatetimeIndex) -> pd.Series:
        if len(index) == 0:
            return pd.Series(dtype=float, name="usdegp")
        rng = np.random.default_rng(self.seed)
        anchors = [(pd.Timestamp(d), level) for d, level in USDEGP_REGIME_ANCHORS]
        anchor_dates = [a[0] for a in anchors]
        anchor_levels = np.array([a[1] for a in anchors])

        # Linear-in-log interpolation between anchors as the deterministic backbone.
        log_levels = np.log(anchor_levels)
        anchor_ords = np.array([d.toordinal() for d in anchor_dates], dtype=float)
        target_ords = np.array([d.toordinal() for d in index], dtype=float)
        backbone_log = np.interp(target_ords, anchor_ords, log_levels)
        backbone = np.exp(backbone_log)

        # Slow drift around backbone (annualized vol applied per day).
        dt = 1.0 / TRADING_DAYS
        drift_vol = self.annual_drift_vol * np.sqrt(dt)
        drift_innov = rng.standard_normal(len(index)) * drift_vol
        drift_log = np.cumsum(drift_innov)

        # AR(1) parallel-market premium (in log terms).
        premium = np.zeros(len(index))
        for t in range(1, len(index)):
            premium[t] = self.premium_phi * premium[t - 1] + self.premium_sigma * np.sqrt(1 - self.premium_phi ** 2) * rng.standard_normal()
        # Snap to the anchor at each anchor date so devaluations remain crisp.
        snap_mask = np.zeros(len(index), dtype=bool)
        for d, _ in anchors:
            if d in index:
                snap_mask[index.get_loc(d)] = True
        # Mute the drift+premium contribution at snap dates.
        drift_log[snap_mask] = 0.0
        premium[snap_mask] = 0.0
        # Reset cumulative drift between anchor segments to zero so the
        # backbone delivers the documented level at each break.
        cum_drift = np.zeros(len(index))
        seg_start = 0
        for t in range(len(index)):
            if snap_mask[t]:
                cum_drift[t] = 0.0
                seg_start = t
            elif t == 0:
                cum_drift[t] = drift_innov[t]
            else:
                cum_drift[t] = cum_drift[t - 1] + drift_innov[t]
        level = backbone * np.exp(cum_drift + premium)
        return pd.Series(level, index=index, name="usdegp")

    def returns(self, index: pd.DatetimeIndex) -> pd.Series:
        level = self.daily_series(index)
        rets = level.pct_change()
        rets.iloc[0] = 0.0
        rets.name = "return"
        return rets.clip(lower=-0.30, upper=0.40)  # cap any single-day move


# ---------------------------------------------------------------------------
# Egypt CPI feed
# ---------------------------------------------------------------------------


# Calibrated to CAPMAS / IMF Egypt CPI year-end YoY observations.
EGYPT_CPI_YOY_SCHEDULE: Dict[int, float] = {
    2014: 0.110, 2015: 0.110, 2016: 0.130,
    2017: 0.295, 2018: 0.140, 2019: 0.090,
    2020: 0.057, 2021: 0.052, 2022: 0.140,
    2023: 0.330, 2024: 0.280, 2025: 0.150,
    2026: 0.120,
}


InflationRegime = Literal["low", "rising", "high", "elevated_falling"]


@dataclass
class EgyptCPIFeed:
    """Daily CPI level + YoY inflation + tactical regime classifier."""

    base_index_value: float = 100.0

    def yoy_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Daily YoY inflation by linear interpolation between annual anchors."""
        if len(index) == 0:
            return pd.Series(dtype=float, name="cpi_yoy")
        years = sorted(EGYPT_CPI_YOY_SCHEDULE.keys())
        anchor_dates = [pd.Timestamp(year=y, month=12, day=31) for y in years]
        anchor_yoy = np.array([EGYPT_CPI_YOY_SCHEDULE[y] for y in years])
        target_ords = np.array([d.toordinal() for d in index], dtype=float)
        anchor_ords = np.array([d.toordinal() for d in anchor_dates], dtype=float)
        yoy = np.interp(target_ords, anchor_ords, anchor_yoy)
        return pd.Series(yoy, index=index, name="cpi_yoy")

    def daily_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Daily CPI level compounded from interpolated daily inflation."""
        yoy = self.yoy_series(index)
        daily_infl = (1.0 + yoy) ** (1.0 / TRADING_DAYS) - 1.0
        level = (1.0 + daily_infl).cumprod() * self.base_index_value
        level.name = "cpi"
        return level

    def real_yield_series(self, tbill_yield: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        yoy = self.yoy_series(index).reindex(tbill_yield.index).fillna(method="ffill")
        real = tbill_yield - yoy
        real.name = "real_yield"
        return real

    def classify_regime(self, day: pd.Timestamp, lookback_days: int = 90) -> InflationRegime:
        """Classify the inflation regime at ``day``.

        ``low``               : YoY < 8 %.
        ``rising``            : 8-15 % and trend up over ``lookback_days``.
        ``high``              : >15 % and trend up.
        ``elevated_falling``  : >15 % but trend down (post-shock disinflation).
        """
        idx = pd.bdate_range(day - pd.Timedelta(days=lookback_days), day)
        s = self.yoy_series(idx)
        if s.empty:
            return "low"
        latest = float(s.iloc[-1])
        trend = float(s.iloc[-1] - s.iloc[0])
        if latest < 0.08:
            return "low"
        if latest < 0.15:
            return "rising" if trend > 0 else "low"
        return "high" if trend > 0 else "elevated_falling"
