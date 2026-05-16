from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import EGX_EQUITY_CAP, OPTIMAL_PORTFOLIO_KEY, PROFILE_BOUNDS, STRATEGIC_PROFILES
from src.strategic.optimization.efficient_frontier import (
    build_capital_market_assumptions,
    optimize_frontier_target_volatility_exact,
)


def test_target_vol_solver_returns_feasible_weights(market_panel, risk_free_rate):
    returns = market_panel.completed_returns
    full_history = pd.DataFrame(
        {asset: market_panel.features[asset]["return"] for asset in returns.columns if "return" in market_panel.features[asset]}
    )
    cma = build_capital_market_assumptions(
        returns,
        expected_returns_forecast={},
        risk_free_rate=risk_free_rate,
        full_history=full_history,
        imputed_fraction=(market_panel.em_result.imputed_fraction if market_panel.em_result else None),
    )
    target_vol = float(STRATEGIC_PROFILES[OPTIMAL_PORTFOLIO_KEY]["target_vol"])
    point = optimize_frontier_target_volatility_exact(
        cma,
        target_vol=target_vol,
        asset_bounds=PROFILE_BOUNDS[OPTIMAL_PORTFOLIO_KEY],
        equity_cap=EGX_EQUITY_CAP[OPTIMAL_PORTFOLIO_KEY],
    )
    w = np.array(list(point.weights.values()), dtype=float)
    assert np.all(np.isfinite(w))
    assert abs(float(w.sum()) - 1.0) <= 1e-4
    assert point.volatility <= target_vol + 0.01
