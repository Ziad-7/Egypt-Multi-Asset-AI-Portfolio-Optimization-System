"""Data loaders and quality controls for the Egyptian multi-asset universe.

Phase-2 upgrades on top of the original audit:

* Replaces the strict ``dropna(how="any")`` panel with an EM-imputed
  *completed* return frame so assets with staggered inception (Gold from
  2024-06, REIT from 2020-12, EGX100 ending 2020-05) contribute to mu and
  Sigma without truncating five years of EGX30/T-Bills history.
* Pulls T-Bill yields from a configurable :mod:`macro_feeds`
  ``TBillYieldFeed`` -- the Ornstein-Uhlenbeck simulator anchored to
  the CBE schedule is the default, exposing real intra-year auction
  volatility instead of a piecewise-constant carry.
* Synthesizes a USD/EGP parallel rate from documented devaluation
  regime breaks; used both as an optional tradable hedge sleeve and as
  the deflator that produces USD-real return series.
* Maintains the previous Geltner unsmoothing for the appraisal-based
  REIT NAV and the +/- 30 % winsorization policy for raw returns.
* Emits a structured ``DataQualityReport`` with imputation diagnostics
  so the operator can see exactly which observations are model-derived.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config.settings import (
    ASSET_FILES,
    INCLUDE_USDEGP_IN_UNIVERSE,
    STRATEGIC_UNIVERSE,
    TBILL_FEED,
    TRADING_DAYS,
    USE_EM_IMPUTATION,
)
from src.data.imputation import EMResult, em_impute_returns
from src.data.macro_feeds import (
    EgyptCPIFeed,
    USDEGPRateFeed,
    _build_tbill_feed,
    yield_to_daily_hpr,
)

logger = logging.getLogger(__name__)


RETURN_OUTLIER_THRESHOLD = 0.30


@dataclass
class DataQualityReport:
    """Structured QA artifact produced alongside the panel."""

    asset_summary: Dict[str, Dict[str, object]] = field(default_factory=dict)
    panel_window: Tuple[pd.Timestamp, pd.Timestamp] | None = None
    aligned_panel_window: Tuple[pd.Timestamp, pd.Timestamp] | None = None
    pairwise_overlap: Dict[str, Dict[str, int]] = field(default_factory=dict)
    warnings: List[Dict[str, str]] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    imputation: Dict[str, object] = field(default_factory=dict)
    macro_feeds: Dict[str, str] = field(default_factory=dict)
    provenance: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "asset_summary": self.asset_summary,
            "panel_window": (
                [str(self.panel_window[0].date()), str(self.panel_window[1].date())]
                if self.panel_window
                else None
            ),
            "aligned_panel_window": (
                [str(self.aligned_panel_window[0].date()), str(self.aligned_panel_window[1].date())]
                if self.aligned_panel_window
                else None
            ),
            "pairwise_overlap": self.pairwise_overlap,
            "warnings": self.warnings,
            "transformations": self.transformations,
            "imputation": self.imputation,
            "macro_feeds": self.macro_feeds,
            "provenance": self.provenance,
        }


@dataclass
class MarketPanel:
    """Aligned multi-asset return panel plus per-asset feature frames."""

    prices: pd.DataFrame
    returns: pd.DataFrame                     # union-index panel; raw observed values + NaN where missing
    completed_returns: pd.DataFrame            # EM-completed panel (or copy of returns when USE_EM_IMPUTATION=False)
    aligned_returns: pd.DataFrame              # strict-overlap panel (kept for tactical / regime use)
    raw_returns: pd.DataFrame                  # per-asset observed-only returns
    features: Dict[str, pd.DataFrame]
    quality: DataQualityReport
    risk_free_daily: pd.Series                 # Egypt T-Bill daily HPR aligned to the universe index
    tbill_yield: pd.Series                     # raw annualized yield (level) used for real-yield calc
    fx_levels: pd.Series                       # synthetic USD/EGP level (EGP per USD)
    fx_returns: pd.Series                      # daily change in USD/EGP level
    cpi_yoy: pd.Series                         # daily YoY inflation rate
    em_result: EMResult | None = None

    @property
    def universe(self) -> List[str]:
        return list(self.completed_returns.columns)


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("Asset CSV must include a 'date' column.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df = df.drop_duplicates(subset="date", keep="last")
    df = df.set_index("date")
    df.index = df.index.tz_localize(None)
    return df.dropna(how="all")


def _warn(report: DataQualityReport, severity: str, code: str, message: str) -> None:
    report.warnings.append({"severity": severity, "code": code, "message": message})


def _winsorize_returns(series: pd.Series, threshold: float = RETURN_OUTLIER_THRESHOLD) -> pd.Series:
    return series.clip(lower=-threshold, upper=threshold)


def _unsmooth_geltner(series: pd.Series, rho: float | None = None) -> pd.Series:
    s = series.dropna().astype(float)
    if len(s) < 30:
        return series
    if rho is None:
        rho = float(s.autocorr(lag=1) or 0.0)
    rho = float(np.clip(rho, 0.0, 0.85))
    if rho < 0.10:
        return series
    unsmoothed = (s - rho * s.shift(1)) / (1.0 - rho)
    return unsmoothed.reindex(series.index)


def _summarize_series(name: str, series: pd.Series) -> Dict[str, object]:
    s = series.dropna()
    if s.empty:
        return {"asset": name, "rows": 0}
    return {
        "asset": name,
        "rows": int(len(s)),
        "start": str(s.index.min().date()),
        "end": str(s.index.max().date()),
        "zero_pct": float((s == 0).mean()),
        "annualized_mean": float(s.mean() * TRADING_DAYS),
        "annualized_vol": float(s.std() * np.sqrt(TRADING_DAYS)),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurt()),
    }


def _series_quality_flags(frame: pd.DataFrame, return_col: str = "return") -> Dict[str, object]:
    if return_col not in frame.columns:
        return {"has_return": False}
    s = pd.to_numeric(frame[return_col], errors="coerce")
    idx = frame.index.to_series().sort_values()
    gaps = idx.diff().dt.days.dropna()
    zero_pct = float((s.fillna(0.0) == 0.0).mean()) if len(s) else 0.0
    stale = bool(gaps.max() > 30) if not gaps.empty else False
    return {
        "has_return": True,
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "max_gap_days": int(gaps.max()) if not gaps.empty else 0,
        "median_gap_days": float(gaps.median()) if not gaps.empty else 0.0,
        "zero_return_pct": zero_pct,
        "stale_series_flag": stale,
    }


def load_market_panel(universe: Tuple[str, ...] | List[str] | None = None) -> MarketPanel:
    """Load all asset files, run quality controls, and return a completed panel.

    The returned ``returns`` is the *observed* union-index panel (NaNs
    where an asset has no data).  ``completed_returns`` is the EM-imputed
    version used by the optimizer; ``aligned_returns`` is the strict-
    overlap subset retained for tactical / regime modules where causal
    alignment is non-negotiable.
    """
    universe = list(universe) if universe is not None else list(STRATEGIC_UNIVERSE)
    report = DataQualityReport()
    report.macro_feeds = {"tbill": TBILL_FEED, "fx": "synthetic", "cpi": "synthetic"}

    features: Dict[str, pd.DataFrame] = {}
    raw_return_series: Dict[str, pd.Series] = {}

    for asset, path in ASSET_FILES.items():
        if not path.exists():
            _warn(report, "critical", "MISSING_ASSET_FILE", f"Missing CSV for {asset}: {path}")
            continue
        df = pd.read_csv(path)
        df = _normalize_frame(df)
        report.provenance[asset] = {
            "file": str(path),
            "source_type": "raw",
            "transformation": [],
            "effective_rows": int(len(df)),
        }

        if asset == "TBills":
            # Always reconstruct via the configured feed so the rest of the
            # platform is decoupled from the raw CSV's pathological columns.
            feed = _build_tbill_feed(TBILL_FEED)
            yield_series = feed.daily_series(df.index)
            df["tbill_yield_raw"] = df.get("annual_yield")
            df["tbill_yield"] = yield_series
            df["return_raw"] = df.get("return")
            df["return"] = yield_to_daily_hpr(yield_series)
            report.transformations.append(
                f"TBills: replaced raw delta-yield with {TBILL_FEED} feed (Egypt CBE anchored)"
            )
            report.provenance[asset]["source_type"] = "synthetic_feed"
            report.provenance[asset]["transformation"].append("tbill_yield_reconstruction")
        elif asset == "EgyptiansRealEstateFund" and "return" in df.columns:
            original = df["return"].copy()
            df["return_raw"] = original
            df["return"] = _unsmooth_geltner(original)
            report.transformations.append(
                "EgyptiansRealEstateFund: Geltner unsmoothing applied to remove appraisal smoothing"
            )
            report.provenance[asset]["transformation"].append("geltner_unsmoothing")

        if "return" in df.columns:
            df["return"] = _winsorize_returns(df["return"].astype(float))
            raw_return_series[asset] = df["return"].dropna()
            report.provenance[asset]["transformation"].append("winsorize_30pct")

        features[asset] = df
        report.asset_summary[asset] = _summarize_series(asset, df.get("return", pd.Series(dtype=float)))
        report.asset_summary[asset]["quality_flags"] = _series_quality_flags(df)
        qf = report.asset_summary[asset]["quality_flags"]
        if qf.get("stale_series_flag"):
            _warn(report, "high", "STALE_SERIES", f"{asset} has stale periods; max gap {qf.get('max_gap_days')} days")
        if qf.get("zero_return_pct", 0.0) > 0.30:
            _warn(
                report,
                "medium",
                "HIGH_ZERO_RETURNS",
                f"{asset} has high zero-return concentration ({qf.get('zero_return_pct'):.2%})",
            )

    if not raw_return_series:
        raise RuntimeError("No usable return series found in any asset CSV.")

    raw_returns_df = pd.DataFrame(raw_return_series).sort_index()

    # Synthetic USD/EGP series on the union index.  Provided to the optimizer
    # only when INCLUDE_USDEGP_IN_UNIVERSE is True; otherwise it is published
    # in the report as a denominator for USD-real metrics.
    union_index = raw_returns_df.index
    fx_feed = USDEGPRateFeed()
    fx_levels = fx_feed.daily_series(union_index)
    fx_returns = fx_feed.returns(union_index)

    if INCLUDE_USDEGP_IN_UNIVERSE:
        raw_returns_df["USDEGP"] = fx_returns.reindex(raw_returns_df.index)
        report.transformations.append(
            "USDEGP: synthetic CIB-GDR-style parallel rate added to active universe"
        )

    active = [a for a in universe if a in raw_returns_df.columns]
    if INCLUDE_USDEGP_IN_UNIVERSE and "USDEGP" not in active:
        active.append("USDEGP")
    if not active:
        raise RuntimeError("None of the requested universe assets are available.")

    union = raw_returns_df[active].copy()
    aligned = union.dropna(how="any")

    if aligned.empty:
        _warn(report, "high", "EMPTY_ALIGNED_PANEL", "No fully-observed dates across the active universe -- aligned panel is empty.")

    em_result: EMResult | None = None
    if USE_EM_IMPUTATION:
        try:
            em_result = em_impute_returns(union)
            completed = em_result.completed[active].copy()
            report.imputation = em_result.to_dict()
            report.transformations.append(
                f"Active universe: EM imputer completed {len(active)} assets in {em_result.iterations} iterations"
            )
            for asset, frac in em_result.imputed_fraction.items():
                if asset in report.provenance:
                    report.provenance[asset]["imputed_fraction"] = float(frac)
                    if float(frac) > 0.50:
                        report.provenance[asset]["source_type"] = "mostly_imputed"
        except Exception as exc:  # pragma: no cover - pathological data only
            logger.exception("EM imputer failed; falling back to aligned panel.")
            _warn(report, "high", "EM_FAILURE", f"EM imputer failed: {exc}; using aligned panel.")
            completed = aligned.copy()
    else:
        completed = aligned.copy()

    completed = completed.sort_index()
    report.panel_window = (
        (completed.index.min(), completed.index.max()) if not completed.empty else None
    )
    report.aligned_panel_window = (
        (aligned.index.min(), aligned.index.max()) if not aligned.empty else None
    )

    overlap = {}
    for a in active:
        overlap[a] = {b: int(union[[a, b]].dropna().shape[0]) for b in active}
    report.pairwise_overlap = overlap

    # Risk-free daily HPR from the chosen T-Bill feed (so optimizer / Sharpe
    # use the same series the platform models as the risk-free asset).
    feed = _build_tbill_feed(TBILL_FEED)
    tbill_yield_universe = feed.daily_series(completed.index)
    risk_free_daily = yield_to_daily_hpr(tbill_yield_universe)
    risk_free_daily.name = "risk_free_daily"

    # CPI YoY series aligned to the same index.
    cpi_yoy = EgyptCPIFeed().yoy_series(completed.index)

    prices = (1.0 + completed).cumprod()

    return MarketPanel(
        prices=prices,
        returns=union,
        completed_returns=completed,
        aligned_returns=aligned[active] if not aligned.empty else aligned,
        raw_returns=raw_returns_df,
        features=features,
        quality=report,
        risk_free_daily=risk_free_daily,
        tbill_yield=tbill_yield_universe,
        fx_levels=fx_levels.reindex(completed.index),
        fx_returns=fx_returns.reindex(completed.index).fillna(0.0),
        cpi_yoy=cpi_yoy,
        em_result=em_result,
    )
