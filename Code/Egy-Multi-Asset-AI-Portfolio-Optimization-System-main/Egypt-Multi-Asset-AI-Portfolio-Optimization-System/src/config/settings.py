"""Project configuration.

Centralizes all paths, asset universes, risk-free schedule, profile risk
budgets and reproducibility knobs.  All hard-coded magic numbers in the
codebase should resolve back to constants in this module so the platform
stays auditable.

Egyptian risk-free schedule
---------------------------
The supplied ``final_tbills.csv`` "return" column is *changes* in the
annualized yield divided by 252.  It is NOT the holding-period return of
a T-Bill (median value 0, ~30 % zero days, +/- 30 % outliers).  Using it
as a return series destroys the carry that fixed income contributes to a
multi-asset portfolio.  We therefore replace it with a piecewise-constant
schedule of Egyptian 91-day T-Bill yields based on Central Bank of Egypt
auctions, then re-derive a clean daily holding-period return inside
``data.loaders``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FINAL_DATA_DIR = DATA_DIR / "final_data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned_data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_SUMMARY_PATH = OUTPUTS_DIR / "model_summary.json"

ASSET_FILES: Dict[str, Path] = {
    "EGX30": FINAL_DATA_DIR / "final_EGX30.csv",
    "EGX100": FINAL_DATA_DIR / "final_EGX100.csv",
    "Gold": FINAL_DATA_DIR / "final_gold.csv",
    "TBills": FINAL_DATA_DIR / "final_tbills.csv",
    "EgyptiansRealEstateFund": FINAL_DATA_DIR / "final_egyptians_real_estate_fund.csv",
}

# Active universe used by the strategic optimizer.
# EGX100 is reinstated alongside EGX30 to capture broad/mid-cap exposure.
# Its post-2020 history is reconstructed by the EM imputer and the optimizer
# enforces a joint EGX30+EGX100 cap (see EGX_EQUITY_CAP) so the two cannot be
# stacked into an over-concentrated equity sleeve.
STRATEGIC_UNIVERSE: Tuple[str, ...] = (
    "EGX30",
    "EGX100",
    "Gold",
    "TBills",
    "EgyptiansRealEstateFund",
)

# Single strategic portfolio label (efficient frontier + Monte Carlo pick one best).
OPTIMAL_PORTFOLIO_KEY = "Optimal"

# Optionally include the synthetic USDEGP sleeve in the optimizer.  Off by
# default because the series is synthesized; flip this to ``True`` to allow
# the optimizer to size a USD hedge sleeve up to ``USDEGP_MAX_WEIGHT``.
INCLUDE_USDEGP_IN_UNIVERSE = False
USDEGP_MAX_WEIGHT = 0.25

# Joint cap on EGX30 + EGX100 to prevent the optimizer from stacking the two
# correlated equity assets into an over-concentrated sleeve.
EGX_EQUITY_CAP: Dict[str, float] = {
    OPTIMAL_PORTFOLIO_KEY: 0.65,
}

# Single strategic profile: risk budget in *annualized standard deviation* terms.
STRATEGIC_PROFILES: Dict[str, Dict[str, float]] = {
    OPTIMAL_PORTFOLIO_KEY: {
        "risk_aversion": 3.0,
        "target_vol": 0.15,
        "max_drawdown_tol": 0.25,
    },
}

# Hard concentration caps and floors enforced inside the optimizer.
PROFILE_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    OPTIMAL_PORTFOLIO_KEY: {
        "EGX30":                   (0.10, 0.50),
        "EGX100":                  (0.04, 0.30),
        "Gold":                    (0.04, 0.35),
        "TBills":                  (0.04, 0.45),
        "EgyptiansRealEstateFund": (0.05, 0.35),
    },
}

# Egyptian 91-day T-Bill yield schedule (annual, decimal).  Sourced from the
# Central Bank of Egypt weekly auctions; encoded as piecewise-constant per year
# to avoid a brittle dependency on intra-day macro feeds.  The values are
# average primary-auction yields for each calendar window.
EGYPT_TBILL_YIELD_SCHEDULE: Dict[Tuple[int, int], float] = {
    (2015, 2015): 0.115,
    (2016, 2016): 0.135,
    (2017, 2017): 0.190,
    (2018, 2018): 0.184,
    (2019, 2019): 0.165,
    (2020, 2020): 0.135,
    (2021, 2021): 0.130,
    (2022, 2022): 0.155,
    (2023, 2023): 0.215,
    (2024, 2024): 0.260,
    (2025, 2026): 0.255,
}

# Annualization conventions.  Egyptian markets observe ~248 trading days/year;
# 252 is retained for cross-comparability with global benchmarks.
TRADING_DAYS = 252
ROLLING_VOL_WINDOW = 20
DEFAULT_FORECAST_BLEND = 0.10  # weight on forward forecast vs history
FORECAST_DEVIATION_CAP_ANNUAL = 0.04  # forecasts can shift mu at most +/- 4 %/yr

# Recent-window risk-free anchor.  Used wherever a single scalar RF rate is
# required (Sharpe, MVO objective).  Falls back to the most recent T-Bill
# schedule entry when the panel cannot resolve a window.
DEFAULT_RISK_FREE_RATE = 0.255

# Backtest evaluation window (Jan 2026–May 2026). Walk-forward training uses full
# history before BACKTEST_START_DATE up to BACKTEST_END_DATE.
BACKTEST_START_DATE = "2026-01-01"
BACKTEST_END_DATE = "2026-05-31"
BACKTEST_REBALANCE_FREQ = "Q"
BACKTEST_TRANSACTION_COST_BPS = 12.0

# Reproducibility.  All random ops in the platform pull from this seed.
RANDOM_SEED = 7
MONTE_CARLO_SIMULATIONS = 5000
MONTE_CARLO_HORIZON_DAYS = TRADING_DAYS

# Macro feed selectors.  ``ou_simulated`` mean-reverts to the CBE schedule but
# overlays auction-cycle volatility; flip to ``static`` to reproduce the legacy
# piecewise-constant behavior.
TBILL_FEED = "ou_simulated"

# EM imputation toggle.  When ``True`` the loader builds a completed return
# panel via Expectation-Maximization so assets with staggered inception (Gold,
# REIT, EGX100) contribute to mu and Sigma without requiring a strict-overlap
# window.
USE_EM_IMPUTATION = True

# Imputation safeguards.  When an asset's imputed_pct exceeds this threshold,
# its expected-return estimate is shrunken toward the cross-sectional mean of
# the *fully observed* assets.
IMPUTED_FRACTION_SHRINK_THRESHOLD = 0.35

# Risk metrics.
VAR_CONFIDENCE_LEVELS: Tuple[float, ...] = (0.95, 0.99)


def get_tbill_yield(year: int) -> float:
    """Return the Egyptian 91-day T-Bill yield for the requested year."""
    for (start, end), value in EGYPT_TBILL_YIELD_SCHEDULE.items():
        if start <= year <= end:
            return value
    return DEFAULT_RISK_FREE_RATE
