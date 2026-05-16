"""Institutional risk engine.

Implements the risk metrics expected from a quant portfolio system:
parametric and historical Value-at-Risk, Conditional VaR (a.k.a.
Expected Shortfall), maximum drawdown, downside-deviation Sortino,
Calmar, beta against an external benchmark, and per-asset marginal /
component risk decomposition.  All routines are deterministic and unit-
testable: no I/O, no random draws.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import TRADING_DAYS


@dataclass
class RiskMetrics:
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    skew: float
    kurtosis: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    historical_var_95: float
    historical_cvar_95: float
    beta: float | None
    tracking_error: float | None
    information_ratio: float | None
    risk_contributions: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return 0.0
    running = cumulative.cummax()
    dd = cumulative / running - 1.0
    return float(dd.min())


def parametric_var(mean: float, std: float, confidence: float) -> float:
    """Closed-form 1-day parametric VaR under Gaussian assumption."""
    z = stats.norm.ppf(1.0 - confidence)
    return float(-(mean + z * std))


def parametric_cvar(mean: float, std: float, confidence: float) -> float:
    """Closed-form 1-day parametric CVaR (a.k.a. Expected Shortfall)."""
    alpha = 1.0 - confidence
    z = stats.norm.ppf(alpha)
    # Standard ES expression: -mu + sigma * phi(z)/alpha
    return float(-mean + std * stats.norm.pdf(z) / alpha)


def historical_var_cvar(returns: pd.Series, confidence: float) -> tuple[float, float]:
    """Empirical historical VaR / CVaR at the requested confidence."""
    s = returns.dropna()
    if s.empty:
        return 0.0, 0.0
    quantile = float(np.quantile(s, 1.0 - confidence))
    var = -quantile
    tail = s[s <= quantile]
    cvar = -float(tail.mean()) if not tail.empty else var
    return var, cvar


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal-risk decomposition: sum equals total portfolio variance."""
    port_var = float(weights @ cov @ weights)
    if port_var <= 0:
        return np.zeros_like(weights)
    marginal = cov @ weights
    contrib = weights * marginal / np.sqrt(port_var)
    return contrib  # in standard-deviation units; sums to portfolio sigma


def downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    s = returns.dropna()
    excess = (s - target).clip(upper=0.0)
    return float(np.sqrt((excess ** 2).mean()) * np.sqrt(TRADING_DAYS))


def compute_portfolio_risk(
    portfolio_returns: pd.Series,
    weights: Dict[str, float] | None = None,
    cov_annualized: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    benchmark_returns: pd.Series | None = None,
) -> RiskMetrics:
    """Aggregate the full risk profile for a portfolio return series.

    Parameters
    ----------
    portfolio_returns
        Daily portfolio return series.
    weights, cov_annualized
        Optional inputs used to compute per-asset risk contributions.
    risk_free_rate
        Annualized risk-free rate (decimal) for Sharpe/Sortino.
    benchmark_returns
        Optional benchmark daily series aligned to ``portfolio_returns``.
    """
    s = portfolio_returns.dropna()
    if s.empty:
        return RiskMetrics(
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            max_drawdown=0.0,
            skew=0.0,
            kurtosis=0.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            historical_var_95=0.0,
            historical_cvar_95=0.0,
            beta=None,
            tracking_error=None,
            information_ratio=None,
            risk_contributions={},
        )

    daily_mean = float(s.mean())
    daily_std = float(s.std())
    ann_return = daily_mean * TRADING_DAYS
    ann_vol = daily_std * np.sqrt(TRADING_DAYS)
    cumulative = (1.0 + s).cumprod()
    mdd = max_drawdown(cumulative)
    dd_dev = downside_deviation(s, target=risk_free_rate / TRADING_DAYS)

    sharpe = (ann_return - risk_free_rate) / (ann_vol + 1e-12)
    sortino = (ann_return - risk_free_rate) / (dd_dev + 1e-12)
    calmar = ann_return / (abs(mdd) + 1e-12) if mdd < 0 else float("nan")

    var95 = parametric_var(daily_mean, daily_std, 0.95)
    var99 = parametric_var(daily_mean, daily_std, 0.99)
    cvar95 = parametric_cvar(daily_mean, daily_std, 0.95)
    cvar99 = parametric_cvar(daily_mean, daily_std, 0.99)
    hvar95, hcvar95 = historical_var_cvar(s, 0.95)

    beta: float | None = None
    tracking_err: float | None = None
    info_ratio: float | None = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        joint = pd.concat([s, benchmark_returns], axis=1, join="inner").dropna()
        if joint.shape[0] > 30:
            cov_pb = float(joint.iloc[:, 0].cov(joint.iloc[:, 1]))
            bench_var = float(joint.iloc[:, 1].var())
            beta = cov_pb / (bench_var + 1e-12)
            active = joint.iloc[:, 0] - joint.iloc[:, 1]
            tracking_err = float(active.std() * np.sqrt(TRADING_DAYS))
            info_ratio = float((active.mean() * TRADING_DAYS) / (tracking_err + 1e-12))

    rc_dict: Dict[str, float] = {}
    if weights is not None and cov_annualized is not None:
        assets = list(weights.keys())
        w = np.array([weights[a] for a in assets], dtype=float)
        cov = cov_annualized.loc[assets, assets].values
        rc = risk_contributions(w, cov)
        port_sigma = float(np.sqrt(w @ cov @ w))
        rc_dict = {
            a: float(rc[i] / (port_sigma + 1e-12)) for i, a in enumerate(assets)
        }

    return RiskMetrics(
        annualized_return=float(ann_return),
        annualized_volatility=float(ann_vol),
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar) if not np.isnan(calmar) else float("nan"),
        max_drawdown=float(mdd),
        skew=float(s.skew()),
        kurtosis=float(s.kurt()),
        var_95=float(var95),
        var_99=float(var99),
        cvar_95=float(cvar95),
        cvar_99=float(cvar99),
        historical_var_95=float(hvar95),
        historical_cvar_95=float(hcvar95),
        beta=beta,
        tracking_error=tracking_err,
        information_ratio=info_ratio,
        risk_contributions=rc_dict,
    )
