"""Monte Carlo path simulation for multi-asset portfolios.

Two engines are provided:

* ``parametric`` -- multivariate Student-t draws (Gaussian when df=inf),
  parameterized from the annualized inputs.  Heavy tails matter for
  Egyptian assets (REIT and Gold both show kurtosis > 8 in the panel).
* ``bootstrap`` -- block-bootstrap of historical daily return rows.
  Preserves cross-sectional correlations, fat tails, and any volatility
  clustering present in the sample.

The simulator returns a structured ``MonteCarloResult`` with terminal-
wealth and drawdown distributions plus annualized risk percentiles, so
the downstream report can publish institutional-grade risk numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import (
    MONTE_CARLO_HORIZON_DAYS,
    MONTE_CARLO_SIMULATIONS,
    RANDOM_SEED,
    TRADING_DAYS,
)


@dataclass
class MonteCarloResult:
    method: str
    horizon_days: int
    simulations: int
    expected_terminal_wealth: float
    median_terminal_wealth: float
    expected_cagr: float
    cagr_5: float
    cagr_25: float
    cagr_75: float
    cagr_95: float
    downside_probability: float        # P(terminal_wealth < 1)
    var_95_annual: float               # 1-yr 95 % VaR (loss as positive number)
    cvar_95_annual: float              # 1-yr 95 % CVaR
    var_99_annual: float
    cvar_99_annual: float
    expected_max_drawdown: float       # mean of path-wise max drawdown (negative)
    worst_max_drawdown: float          # 5th percentile drawdown
    target_breach_probability: float   # P(drawdown < target)
    target_drawdown: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _path_max_drawdown(path: np.ndarray) -> float:
    running_max = np.maximum.accumulate(path)
    drawdown = path / running_max - 1.0
    return float(drawdown.min())


def _summarize(
    method: str,
    portfolio_paths: np.ndarray,
    horizon_days: int,
    drawdowns: np.ndarray,
    target_drawdown: float,
) -> MonteCarloResult:
    terminal = portfolio_paths[:, -1]
    annual_factor = TRADING_DAYS / max(horizon_days, 1)
    cagr = terminal ** annual_factor - 1.0

    # 1-year VaR/CVaR derived from terminal-wealth losses.
    losses = 1.0 - terminal
    var95 = float(np.quantile(losses, 0.95))
    var99 = float(np.quantile(losses, 0.99))
    cvar95 = float(losses[losses >= var95].mean()) if (losses >= var95).any() else var95
    cvar99 = float(losses[losses >= var99].mean()) if (losses >= var99).any() else var99

    return MonteCarloResult(
        method=method,
        horizon_days=int(horizon_days),
        simulations=int(portfolio_paths.shape[0]),
        expected_terminal_wealth=float(terminal.mean()),
        median_terminal_wealth=float(np.median(terminal)),
        expected_cagr=float(np.median(cagr)),
        cagr_5=float(np.quantile(cagr, 0.05)),
        cagr_25=float(np.quantile(cagr, 0.25)),
        cagr_75=float(np.quantile(cagr, 0.75)),
        cagr_95=float(np.quantile(cagr, 0.95)),
        downside_probability=float(np.mean(terminal < 1.0)),
        var_95_annual=float(var95),
        cvar_95_annual=float(cvar95),
        var_99_annual=float(var99),
        cvar_99_annual=float(cvar99),
        expected_max_drawdown=float(np.mean(drawdowns)),
        worst_max_drawdown=float(np.quantile(drawdowns, 0.05)),
        target_breach_probability=float(np.mean(drawdowns < target_drawdown)),
        target_drawdown=float(target_drawdown),
    )


def parametric_monte_carlo(
    weights: Dict[str, float],
    expected_returns_annual: Dict[str, float],
    cov_annual: pd.DataFrame,
    horizon_days: int = MONTE_CARLO_HORIZON_DAYS,
    n_simulations: int = MONTE_CARLO_SIMULATIONS,
    df: float = 6.0,
    target_drawdown: float = -0.20,
    seed: int = RANDOM_SEED,
) -> MonteCarloResult:
    """Multivariate Student-t Monte Carlo with annualized inputs.

    df ~ 6 produces tails materially fatter than Gaussian and matches the
    sample kurtosis observed in the active Egyptian universe.  Pass
    ``df=np.inf`` to fall back to Gaussian.
    """
    assets = list(weights.keys())
    w = np.array([weights[a] for a in assets], dtype=float)
    mu_daily = np.array([expected_returns_annual[a] for a in assets], dtype=float) / TRADING_DAYS
    cov_daily = cov_annual.loc[assets, assets].values / TRADING_DAYS
    cov_daily = 0.5 * (cov_daily + cov_daily.T) + np.eye(len(assets)) * 1e-12

    rng = np.random.default_rng(seed)
    n_assets = len(assets)

    if not np.isfinite(df):
        # Multivariate Gaussian
        draws = rng.multivariate_normal(mu_daily, cov_daily, size=(n_simulations, horizon_days))
    else:
        # Multivariate Student-t: scale a Gaussian draw by sqrt(df / chi2)
        gaussians = rng.multivariate_normal(np.zeros(n_assets), cov_daily, size=(n_simulations, horizon_days))
        chi2 = rng.chisquare(df, size=(n_simulations, horizon_days))[..., None]
        scale = np.sqrt(df / np.maximum(chi2, 1e-12))
        # Rescale so the *unconditional* covariance still matches cov_daily.
        # Multivariate-t with df > 2 has cov = (df / (df - 2)) * Sigma, hence
        # divide draws by sqrt(df / (df - 2)).
        if df > 2:
            scale = scale / np.sqrt(df / (df - 2))
        draws = mu_daily + gaussians * scale

    portfolio_returns = draws @ w  # shape (n_sims, horizon_days)
    portfolio_paths = np.cumprod(1.0 + portfolio_returns, axis=1)
    portfolio_paths = np.clip(portfolio_paths, 1e-9, None)
    drawdowns = np.array([_path_max_drawdown(p) for p in portfolio_paths])

    return _summarize("parametric_student_t", portfolio_paths, horizon_days, drawdowns, target_drawdown)


def bootstrap_monte_carlo(
    weights: Dict[str, float],
    historical_returns: pd.DataFrame,
    horizon_days: int = MONTE_CARLO_HORIZON_DAYS,
    n_simulations: int = MONTE_CARLO_SIMULATIONS,
    block_size: int = 10,
    target_drawdown: float = -0.20,
    seed: int = RANDOM_SEED,
) -> MonteCarloResult:
    """Block-bootstrap historical daily returns to preserve dependence."""
    assets = list(weights.keys())
    panel = historical_returns[assets].dropna(how="any").values
    if panel.shape[0] < block_size + 5:
        raise ValueError("Not enough historical observations for bootstrap.")
    w = np.array([weights[a] for a in assets], dtype=float)

    rng = np.random.default_rng(seed)
    n_obs = panel.shape[0]
    n_blocks = int(np.ceil(horizon_days / block_size))

    sims = np.empty((n_simulations, horizon_days, len(assets)), dtype=float)
    for s in range(n_simulations):
        starts = rng.integers(0, n_obs - block_size + 1, size=n_blocks)
        blocks = np.concatenate([panel[start:start + block_size] for start in starts], axis=0)
        sims[s] = blocks[:horizon_days]

    portfolio_returns = sims @ w
    portfolio_paths = np.cumprod(1.0 + portfolio_returns, axis=1)
    portfolio_paths = np.clip(portfolio_paths, 1e-9, None)
    drawdowns = np.array([_path_max_drawdown(p) for p in portfolio_paths])
    return _summarize("block_bootstrap", portfolio_paths, horizon_days, drawdowns, target_drawdown)


def run_monte_carlo(
    weights: Dict[str, float],
    expected_returns_annual: Dict[str, float],
    cov_annual: pd.DataFrame,
    historical_returns: pd.DataFrame | None = None,
    horizon_days: int = MONTE_CARLO_HORIZON_DAYS,
    n_simulations: int = MONTE_CARLO_SIMULATIONS,
    target_drawdown: float = -0.20,
    seed: int = RANDOM_SEED,
) -> Dict[str, MonteCarloResult]:
    """Run both parametric and (when feasible) block-bootstrap simulations."""
    out: Dict[str, MonteCarloResult] = {}
    out["parametric"] = parametric_monte_carlo(
        weights=weights,
        expected_returns_annual=expected_returns_annual,
        cov_annual=cov_annual,
        horizon_days=horizon_days,
        n_simulations=n_simulations,
        target_drawdown=target_drawdown,
        seed=seed,
    )
    if historical_returns is not None and historical_returns.dropna(how="any").shape[0] > 30:
        try:
            out["bootstrap"] = bootstrap_monte_carlo(
                weights=weights,
                historical_returns=historical_returns,
                horizon_days=horizon_days,
                n_simulations=n_simulations,
                target_drawdown=target_drawdown,
                seed=seed + 1,
            )
        except Exception:
            pass
    return out
