from __future__ import annotations

from src.config.settings import EGX_EQUITY_CAP, OPTIMAL_PORTFOLIO_KEY, PROFILE_BOUNDS
from src.strategic.allocation.engine import build_strategic_allocations


def test_profile_constraints_enforced(market_panel, risk_free_rate):
    outputs, _ = build_strategic_allocations(
        market_panel.completed_returns,
        market_panel.features,
        risk_free_rate=risk_free_rate,
        imputed_fraction=(market_panel.em_result.imputed_fraction if market_panel.em_result else None),
    )
    for profile, out in outputs.items():
        assert abs(sum(out.weights.values()) - 1.0) <= 1e-4
        if profile == OPTIMAL_PORTFOLIO_KEY:
            # Tangency max Sharpe: mandate bounds are advisory only.
            continue
        bounds = PROFILE_BOUNDS[profile]
        for asset, (lb, ub) in bounds.items():
            w = out.weights.get(asset, 0.0)
            assert lb - 1e-6 <= w <= ub + 1e-6, f"{profile}::{asset}={w} outside [{lb}, {ub}]"
        eq_cap = EGX_EQUITY_CAP[profile]
        eq_weight = out.weights.get("EGX30", 0.0) + out.weights.get("EGX100", 0.0)
        assert eq_weight <= eq_cap + 1e-6, f"{profile} equity cap breach {eq_weight} > {eq_cap}"
