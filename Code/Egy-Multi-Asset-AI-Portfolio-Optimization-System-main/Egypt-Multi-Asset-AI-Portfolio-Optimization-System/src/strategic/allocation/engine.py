"""Strategic allocation engine.

Coordinates forecasting, optimization, the risk engine, and Monte Carlo
into a single strategic output.  For ``OPTIMAL_PORTFOLIO_KEY`` the engine
uses the **tangency portfolio**: long-only maximum Sharpe with no
per-asset mandate bounds (same construction as ``reference_portfolios.
maximum_sharpe``).  Other profiles, if added later, use constrained
frontier optimization.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.api.schemas import StrategicAllocationOutput
from src.config.settings import (
    DEFAULT_RISK_FREE_RATE,
    EGX_EQUITY_CAP,
    IMPUTED_FRACTION_SHRINK_THRESHOLD,
    OPTIMAL_PORTFOLIO_KEY,
    PROFILE_BOUNDS,
    STRATEGIC_PROFILES,
)
from src.strategic.forecasting.regression import forecast_expected_returns
from src.strategic.macro.factor_model import build_macro_signal
from src.strategic.optimization.efficient_frontier import (
    build_capital_market_assumptions,
    build_frontier_diagnostics,
    optimize_equal_risk_contribution,
    optimize_frontier_target_volatility_exact,
    optimize_max_sharpe,
    optimize_min_variance,
)
from src.strategic.risk_models.monte_carlo import run_monte_carlo
from src.strategic.risk_models.risk_engine import compute_portfolio_risk


def _assert_profile_constraints(profile: str, weights: Dict[str, float]) -> Dict[str, object]:
    bounds = PROFILE_BOUNDS.get(profile, {})
    diagnostics = {"asset_bounds_ok": True, "equity_cap_ok": True, "violations": []}
    for asset, (lb, ub) in bounds.items():
        w = float(weights.get(asset, 0.0))
        if w < lb - 1e-6 or w > ub + 1e-6:
            diagnostics["asset_bounds_ok"] = False
            diagnostics["violations"].append(
                f"{asset} weight {w:.4f} outside [{lb:.4f}, {ub:.4f}]"
            )
    eq_cap = EGX_EQUITY_CAP.get(profile)
    if eq_cap is not None:
        eq_weight = float(weights.get("EGX30", 0.0) + weights.get("EGX100", 0.0))
        if eq_weight > float(eq_cap) + 1e-6:
            diagnostics["equity_cap_ok"] = False
            diagnostics["violations"].append(
                f"EGX30+EGX100 {eq_weight:.4f} exceeds cap {float(eq_cap):.4f}"
            )
    diagnostics["sum_weights"] = float(sum(weights.values()))
    if abs(diagnostics["sum_weights"] - 1.0) > 1e-4:
        diagnostics["violations"].append(f"weights sum {diagnostics['sum_weights']:.6f} != 1.0")
    return diagnostics


def _next_quarter_rebalance(last_date: pd.Timestamp) -> date:
    quarter_end_month = ((last_date.month - 1) // 3 + 1) * 3
    quarter_end = pd.Timestamp(year=last_date.year, month=quarter_end_month, day=1) + pd.offsets.MonthEnd(0)
    if last_date.date() >= quarter_end.date():
        quarter_end = quarter_end + pd.offsets.QuarterEnd()
    return quarter_end.date()


def _profile_rebalance_note(profile: str, macro_inflation: float, macro_rates: float) -> str:
    base = f"Quarterly strategic rebalance for {profile} profile."
    if macro_inflation > 0.0005:
        return base + " Inflation pressure positive: tilt toward Gold and short-duration T-Bills."
    if macro_rates > 0.0005:
        return base + " Real-yield pressure rising: keep TBills near upper bound."
    return base + " Macro environment neutral: keep risk budget on profile target."


def build_strategic_allocations(
    returns: pd.DataFrame,
    features: Dict[str, pd.DataFrame],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    imputed_fraction: Dict[str, float] | None = None,
) -> Tuple[Dict[str, StrategicAllocationOutput], dict]:
    forecasts = forecast_expected_returns(features)
    macro = build_macro_signal(features)

    universe = list(returns.columns)
    panel = returns[universe].copy()

    full_history = pd.DataFrame(
        {asset: features[asset]["return"] for asset in universe if "return" in features.get(asset, pd.DataFrame()).columns}
    )

    cma = build_capital_market_assumptions(
        panel,
        expected_returns_forecast={a: forecasts.expected_returns[a] for a in universe if a in forecasts.expected_returns},
        risk_free_rate=risk_free_rate,
        full_history=full_history,
        imputed_fraction=imputed_fraction,
        imputed_shrink_threshold=IMPUTED_FRACTION_SHRINK_THRESHOLD,
    )
    cov_df = cma.cov_dataframe()

    last_date = panel.dropna(how="all").index.max()
    next_rebalance = _next_quarter_rebalance(last_date)

    outputs: Dict[str, StrategicAllocationOutput] = {}
    forecast_conf = float(np.median(list(forecasts.confidence.values()))) if forecasts.confidence else 0.2

    for profile, cfg in STRATEGIC_PROFILES.items():
        target_vol = float(cfg["target_vol"])
        max_dd_tol = float(cfg.get("max_drawdown_tol", -0.20))
        bounds = PROFILE_BOUNDS.get(profile)
        eq_cap = EGX_EQUITY_CAP.get(profile)

        if profile == OPTIMAL_PORTFOLIO_KEY:
            # Tangency: max Sharpe on long-only simplex (matches frontier reference "maximum_sharpe").
            point = optimize_max_sharpe(cma, weight_cap=1.0, asset_bounds=None, equity_cap=None)
            method = "tangency_max_sharpe"
        else:
            point = optimize_frontier_target_volatility_exact(
                cma,
                target_vol=target_vol,
                weight_cap=1.0,
                asset_bounds=bounds,
                equity_cap=eq_cap,
            )
            method = "mvo_frontier_target_vol_exact"

        mc = run_monte_carlo(
            weights=point.weights,
            expected_returns_annual={a: float(cma.expected_returns[i]) for i, a in enumerate(cma.assets)},
            cov_annual=cov_df,
            historical_returns=panel,
            target_drawdown=-abs(max_dd_tol),
        )
        mc_dict = {k: v.to_dict() for k, v in mc.items()}

        constraint_diag = _assert_profile_constraints(profile, point.weights)
        if profile == OPTIMAL_PORTFOLIO_KEY:
            constraint_diag["tangency_portfolio"] = True
            constraint_diag["note"] = (
                "PROFILE_BOUNDS and EGX_EQUITY_CAP are advisory for this profile; weights are unconstrained tangency max Sharpe."
            )
        elif constraint_diag["violations"]:
            raise RuntimeError(
                f"Constraint violations for profile {profile}: {constraint_diag['violations']}"
            )

        param_d = mc_dict.get("parametric") or {}
        downside = float(param_d.get("downside_probability", 0.5))
        confidence = float(np.clip(0.55 * (1.0 - downside) + 0.45 * forecast_conf, 0.05, 0.98))

        weights_series = pd.Series(point.weights).reindex(panel.columns).fillna(0.0)
        portfolio_returns = panel.fillna(0.0).mul(weights_series, axis=1).sum(axis=1)

        risk = compute_portfolio_risk(
            portfolio_returns,
            weights=point.weights,
            cov_annualized=cov_df,
            risk_free_rate=risk_free_rate,
        )

        outputs[profile] = StrategicAllocationOutput(
            profile=profile,
            weights={a: round(float(w), 4) for a, w in point.weights.items()},
            expected_return=round(point.expected_return, 4),
            expected_volatility=round(point.volatility, 4),
            sharpe=round(point.sharpe, 4),
            confidence=round(confidence, 4),
            rebalancing_note=_profile_rebalance_note(profile, macro.inflation_pressure, macro.rates_pressure),
            rebalance_frequency="quarterly",
            next_rebalance_date=str(next_rebalance),
            risk_contributions={a: round(v, 4) for a, v in point.risk_contributions.items()},
            diversification_ratio=round(point.diversification_ratio, 4),
            risk_metrics=risk.to_dict(),
            monte_carlo=mc_dict,
            optimization_method=method,
        )
        outputs[profile].risk_metrics["constraint_diagnostics"] = constraint_diag

    diagnostics = build_frontier_diagnostics(cma, weight_cap=1.0, asset_bounds=None, equity_cap=None)
    diagnostics["optimal_policy"] = {
        "key": OPTIMAL_PORTFOLIO_KEY,
        "construction": "tangency_max_sharpe",
        "matches_reference_key": "maximum_sharpe",
    }
    # Use the EM-completed panel for the published correlation matrix; falls
    # back to pairwise corr() if some columns are still all-NaN.
    diagnostics["correlation_matrix"] = panel.corr().round(6).to_dict()
    diagnostics["forecasts"] = {
        "expected_returns_daily": forecasts.expected_returns,
        "confidence": forecasts.confidence,
        "sample_sizes": forecasts.sample_sizes,
        "model_predictions_63d": forecasts.model_predictions_63d,
        "model_metrics_r2": forecasts.model_metrics,
        "model_diagnostics": forecasts.model_diagnostics,
        "best_model": forecasts.best_model,
    }
    diagnostics["macro"] = {
        "trend_score": macro.trend_score,
        "inflation_pressure": macro.inflation_pressure,
        "rates_pressure": macro.rates_pressure,
    }

    reference = {
        "minimum_variance": _serialize_point(optimize_min_variance(cma, weight_cap=1.0, asset_bounds=None, equity_cap=None)),
        "maximum_sharpe": _serialize_point(optimize_max_sharpe(cma, weight_cap=1.0, asset_bounds=None, equity_cap=None)),
        "equal_risk_contribution": _serialize_point(
            optimize_equal_risk_contribution(cma, weight_cap=1.0, asset_bounds=None, equity_cap=None)
        ),
    }
    diagnostics["reference_portfolios"] = reference

    return outputs, diagnostics


def _serialize_point(point) -> dict:
    return {
        "weights": {a: round(float(w), 4) for a, w in point.weights.items()},
        "expected_return": round(point.expected_return, 4),
        "volatility": round(point.volatility, 4),
        "sharpe": round(point.sharpe, 4),
        "objective": point.objective,
        "diversification_ratio": round(point.diversification_ratio, 4),
        "risk_contributions": {a: round(float(v), 4) for a, v in point.risk_contributions.items()},
    }
