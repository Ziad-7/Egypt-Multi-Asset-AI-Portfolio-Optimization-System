from __future__ import annotations

import pandas as pd

from src.config.settings import STRATEGIC_PROFILES
from src.simulation.backtest import run_backtest


def test_backtest_cumulative_accounting_consistent(market_panel, risk_free_rate):
    result = run_backtest(
        market_panel.completed_returns,
        market_panel.features,
        risk_free_rate=risk_free_rate,
        include_sp500=False,
        fx_levels=market_panel.fx_levels,
    )
    assert "error" not in result, result.get("error")
    daily_dates = pd.to_datetime(result["daily_dates"])
    cum = result["daily_cumulative_returns"]
    for profile in STRATEGIC_PROFILES:
        assert profile in cum
        s = pd.Series(cum[profile], index=daily_dates).dropna()
        if s.empty:
            continue
        reported_total = float(result["portfolios"][profile]["metrics"]["total_return"])
        computed_total = float(s.iloc[-1])
        assert abs(reported_total - computed_total) < 2e-3
