from __future__ import annotations

import pandas as pd

from src.simulation.backtest import run_backtest


def test_backtest_walkforward_has_no_train_test_overlap(market_panel, risk_free_rate):
    result = run_backtest(
        market_panel.completed_returns,
        market_panel.features,
        risk_free_rate=risk_free_rate,
        include_sp500=False,
        fx_levels=market_panel.fx_levels,
    )
    assert "error" not in result, result.get("error")
    checks = result.get("leakage_checks", [])
    assert checks, "No leakage checks produced by walk-forward engine."
    for check in checks:
        assert check.get("overlap") is False
        train_end = pd.Timestamp(check["train_end"])
        test_start = pd.Timestamp(check["test_start"])
        assert train_end < test_start, f"Leakage at {check['rebalance_date']}: {train_end} !< {test_start}"
