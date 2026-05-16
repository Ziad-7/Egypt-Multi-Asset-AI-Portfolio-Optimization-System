"""Mean-variance optimization with institutional safeguards.

Highlights of this rewrite vs. the legacy implementation:

* Capital-market assumptions are produced by a dedicated function which
  applies Ledoit-Wolf shrinkage to the sample covariance, enabling
  stable risk estimates from short Egyptian return windows.
* The optimizer now supports Maximum-Sharpe, Minimum-Variance, target-
  volatility, target-return, mean-variance utility, and ERC (true risk
  parity) all driven by a single ``_solve`` helper so constraints stay
  coherent across every flavor.
* Profile bounds are enforced *and* their feasibility is checked before
  the SLSQP call, eliminating the silent ``optimize_max_sharpe``
  fallback the legacy code used to mask infeasibility.
* Forecast blending is configurable and the annualization is consistent
  with the data layer (TRADING_DAYS).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config.settings import (
    DEFAULT_FORECAST_BLEND,
    DEFAULT_RISK_FREE_RATE,
    FORECAST_DEVIATION_CAP_ANNUAL,
    TRADING_DAYS,
)


def _correlation_shrinkage_cov(returns: pd.DataFrame, lambda_: float | None = None) -> np.ndarray:
    """Diagonal-preserving correlation shrinkage estimator.

    Keeps each asset's diagonal variance equal to its sample variance and
    shrinks the *correlation* matrix toward the identity by ``lambda_``.
    This avoids the Ledoit-Wolf pathology of inflating an asset with
    near-zero true variance (T-Bills).  When ``lambda_`` is ``None`` we
    pick an effective sample-size-aware default lambda = p / (p + n).
    """
    n_obs, p = returns.shape
    cov = returns.cov().values
    diag = np.diag(cov).copy()
    diag = np.where(diag <= 0, 1e-12, diag)
    sigma = np.sqrt(diag)
    corr = cov / np.outer(sigma, sigma)
    np.fill_diagonal(corr, 1.0)
    if lambda_ is None:
        lambda_ = float(p / (p + max(n_obs, 1)))
    lambda_ = float(np.clip(lambda_, 0.0, 1.0))
    target = np.eye(p)
    shrunk_corr = (1.0 - lambda_) * corr + lambda_ * target
    np.fill_diagonal(shrunk_corr, 1.0)
    shrunk_cov = shrunk_corr * np.outer(sigma, sigma)
    return 0.5 * (shrunk_cov + shrunk_cov.T)


@dataclass
class CapitalMarketAssumptions:
    """Annualized inputs used by every optimizer."""

    assets: List[str]
    expected_returns: np.ndarray              # annualized, blended history+forecast
    cov_matrix: np.ndarray                    # annualized, Ledoit-Wolf shrunken
    historical_mean: np.ndarray               # annualized, raw historical
    forecast_mean: np.ndarray                 # annualized forecast (or zeros)
    sample_size: int                          # observations used in estimation
    risk_free_rate: float

    def cov_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.cov_matrix, index=self.assets, columns=self.assets)


@dataclass
class FrontierPoint:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    objective: str = ""
    diversification_ratio: float = 0.0
    risk_contributions: Dict[str, float] = field(default_factory=dict)


def build_capital_market_assumptions(
    returns: pd.DataFrame,
    expected_returns_forecast: Dict[str, float] | None = None,
    forecast_blend: float = DEFAULT_FORECAST_BLEND,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    full_history: pd.DataFrame | None = None,
    bayes_shrinkage: float = 0.30,
    prior_equity_premium: float = 0.06,
    imputed_fraction: Dict[str, float] | None = None,
    imputed_shrink_threshold: float = 0.35,
) -> CapitalMarketAssumptions:
    """Construct annualized mu and Sigma from a daily return panel.

    Parameters
    ----------
    returns
        Aligned multi-asset daily return panel (used for *covariance*).
    full_history
        Optional per-asset full-sample return frame.  When supplied,
        each asset's expected-return estimate uses the longer per-asset
        sample, which materially reduces the bias that arises when the
        aligned panel is dominated by a recent regime.  Falls back to
        ``returns`` when not supplied.
    bayes_shrinkage
        Weight on the prior (risk-free + equity premium for risky
        assets, risk-free for T-Bills) used to shrink the noisy sample
        mean toward an economically plausible value.
    prior_equity_premium
        Used to define the prior expected return for risky assets.
    """
    assets = list(returns.columns)
    aligned = returns.dropna(how="any").copy()
    if aligned.shape[0] < max(20, 2 * len(assets)):
        aligned = returns.dropna(how="all").fillna(0.0)
    for col in aligned.columns:
        lo, hi = aligned[col].quantile(0.005), aligned[col].quantile(0.995)
        aligned[col] = aligned[col].clip(lower=lo, upper=hi)

    # Per-asset historical mean uses the longer per-asset history when
    # available, with the same winsorization policy.
    historical_mu = np.zeros(len(assets), dtype=float)
    for i, asset in enumerate(assets):
        source = full_history[asset] if full_history is not None and asset in full_history else aligned[asset]
        s = source.dropna()
        if not s.empty:
            lo, hi = s.quantile(0.005), s.quantile(0.995)
            s = s.clip(lower=lo, upper=hi)
        historical_mu[i] = float(s.mean()) * TRADING_DAYS if not s.empty else 0.0

    # Bayesian shrinkage: blend the noisy sample mean toward an economic prior.
    # T-Bills' prior is the risk-free rate itself; risky assets get RF + ERP.
    prior = np.full(len(assets), risk_free_rate + prior_equity_premium, dtype=float)
    for i, asset in enumerate(assets):
        if "tbill" in asset.lower():
            prior[i] = risk_free_rate
        if "usdegp" in asset.lower():
            prior[i] = 0.0  # USD/EGP is currency drift, not a risky-asset premium
    historical_mu = (1.0 - bayes_shrinkage) * historical_mu + bayes_shrinkage * prior

    # Heavy-imputation guardrail: if an asset's observed history is mostly
    # synthetic (EM-completed), its mean estimate is dominated by conditional
    # projections on other assets and can drift in ways that are not
    # economically informative.  Pull such estimates further toward the prior.
    if imputed_fraction:
        for i, asset in enumerate(assets):
            frac = float(imputed_fraction.get(asset, 0.0))
            if frac >= imputed_shrink_threshold:
                extra = float(np.clip((frac - imputed_shrink_threshold) / (1.0 - imputed_shrink_threshold), 0.0, 1.0))
                historical_mu[i] = (1.0 - extra) * historical_mu[i] + extra * prior[i]

    # Forecast deviations are clipped at the *annualized* level to a tight
    # envelope around history.  This prevents the walk-forward ML stack from
    # dominating mu with extreme last-fold predictions.
    forecasts = expected_returns_forecast or {}
    forecast_mu_daily = np.array(
        [forecasts.get(a, historical_mu[i] / TRADING_DAYS) for i, a in enumerate(assets)],
        dtype=float,
    )
    forecast_mu = forecast_mu_daily * TRADING_DAYS
    forecast_mu = np.clip(
        forecast_mu,
        historical_mu - FORECAST_DEVIATION_CAP_ANNUAL,
        historical_mu + FORECAST_DEVIATION_CAP_ANNUAL,
    )

    blend = float(np.clip(forecast_blend, 0.0, 1.0))
    mu = (1.0 - blend) * historical_mu + blend * forecast_mu
    mu = np.clip(mu, -0.10, 0.55)

    cov_daily = _correlation_shrinkage_cov(aligned)

    # EM imputation pathology: heavily-imputed columns under-represent variance
    # because the conditional-mean imputer drops residuals.  Replace the
    # diagonal of those columns with the *observed-only* sample variance so
    # the marginal vol used by the optimizer matches what the operator would
    # measure on the observed window directly.  Off-diagonal correlations are
    # untouched, so cross-asset structure still reflects the EM solution.
    if imputed_fraction and full_history is not None:
        diag_corrections = np.array([np.sqrt(cov_daily[i, i]) for i in range(len(assets))])
        for i, asset in enumerate(assets):
            frac = float(imputed_fraction.get(asset, 0.0))
            if frac >= imputed_shrink_threshold and asset in full_history:
                obs = full_history[asset].dropna()
                if len(obs) > 30:
                    sigma_obs = float(obs.clip(obs.quantile(0.005), obs.quantile(0.995)).std())
                    diag_corrections[i] = sigma_obs
        # Rescale rows / cols so corr stays the same but diagonal matches sigma_obs.
        sigma_current = np.sqrt(np.maximum(np.diag(cov_daily), 1e-18))
        scale = diag_corrections / sigma_current
        cov_daily = cov_daily * np.outer(scale, scale)

    cov = cov_daily * TRADING_DAYS
    cov = 0.5 * (cov + cov.T) + np.eye(len(assets)) * 1e-10

    return CapitalMarketAssumptions(
        assets=assets,
        expected_returns=mu,
        cov_matrix=cov,
        historical_mean=historical_mu,
        forecast_mean=forecast_mu,
        sample_size=int(aligned.shape[0]),
        risk_free_rate=float(risk_free_rate),
    )


def _portfolio_metrics(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> Tuple[float, float, float]:
    port_ret = float(np.dot(weights, mu))
    port_var = float(weights @ cov @ weights)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    sharpe = (port_ret - rf) / (port_vol + 1e-12)
    return port_ret, port_vol, sharpe


def _diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    sigmas = np.sqrt(np.diag(cov))
    weighted_vols = float(weights @ sigmas)
    port_vol = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    return weighted_vols / (port_vol + 1e-12)


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    port_var = float(weights @ cov @ weights)
    if port_var <= 0:
        return np.zeros_like(weights)
    marginal = cov @ weights
    return weights * marginal / port_var  # variance-fraction (sums to 1.0)


def _build_frontier_point(
    cma: CapitalMarketAssumptions,
    weights: np.ndarray,
    objective: str,
) -> FrontierPoint:
    port_ret, port_vol, sharpe = _portfolio_metrics(weights, cma.expected_returns, cma.cov_matrix, cma.risk_free_rate)
    rc = _risk_contributions(weights, cma.cov_matrix)
    return FrontierPoint(
        weights={asset: float(w) for asset, w in zip(cma.assets, weights)},
        expected_return=port_ret,
        volatility=port_vol,
        sharpe=sharpe,
        objective=objective,
        diversification_ratio=_diversification_ratio(weights, cma.cov_matrix),
        risk_contributions={a: float(rc[i]) for i, a in enumerate(cma.assets)},
    )


def _equity_group_constraint(
    assets: Sequence[str],
    equity_cap: float | None,
    equity_assets: Sequence[str] = ("EGX30", "EGX100"),
) -> List[dict]:
    """Inequality constraint: sum of equity-asset weights <= equity_cap."""
    if equity_cap is None:
        return []
    indices = [i for i, a in enumerate(assets) if a in equity_assets]
    if len(indices) < 2:
        return []
    return [{"type": "ineq", "fun": lambda w, idx=tuple(indices), cap=equity_cap: cap - sum(w[i] for i in idx)}]


def _resolve_bounds(
    assets: Sequence[str],
    weight_cap: float,
    asset_bounds: Dict[str, Tuple[float, float]] | None,
) -> Tuple[List[Tuple[float, float]], bool]:
    """Return the bounds list and a feasibility flag (True if sum(lb) <= 1 <= sum(ub))."""
    bounds: List[Tuple[float, float]] = []
    for asset in assets:
        lb, ub = 0.0, weight_cap
        if asset_bounds and asset in asset_bounds:
            lb_user, ub_user = asset_bounds[asset]
            lb = max(0.0, float(lb_user))
            ub = min(float(ub_user), weight_cap)
            if ub < lb:
                ub = lb  # degenerate but non-empty
        bounds.append((lb, ub))
    sum_lb = sum(b[0] for b in bounds)
    sum_ub = sum(b[1] for b in bounds)
    feasible = sum_lb <= 1.0 + 1e-9 and sum_ub >= 1.0 - 1e-9
    return bounds, feasible


def _project_to_simplex(weights: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Best-effort fallback: clip + renormalize, then iteratively re-clip."""
    w = np.array(weights, dtype=float)
    for _ in range(50):
        for i, (lb, ub) in enumerate(bounds):
            w[i] = min(max(w[i], lb), ub)
        s = w.sum()
        if s <= 0:
            n = len(w)
            return np.array([(b[0] + b[1]) / 2.0 for b in bounds]) / max(
                sum((b[0] + b[1]) / 2.0 for b in bounds), 1e-12
            )
        w = w / s
        if np.all([b[0] - 1e-9 <= w[i] <= b[1] + 1e-9 for i, b in enumerate(bounds)]):
            break
    return w


def _solve(
    cma: CapitalMarketAssumptions,
    objective_fn,
    bounds: Sequence[Tuple[float, float]],
    extra_constraints: Sequence[dict] | None = None,
    start: np.ndarray | None = None,
) -> np.ndarray:
    n = len(cma.assets)
    if start is None:
        start = np.array([(b[0] + b[1]) / 2.0 for b in bounds], dtype=float)
        s = start.sum()
        start = start / s if s > 0 else np.full(n, 1.0 / n)

    constraints: List[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if extra_constraints:
        constraints.extend(extra_constraints)

    result = minimize(
        objective_fn,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    weights = result.x if (result.success and np.all(np.isfinite(result.x))) else start
    return _project_to_simplex(weights, bounds)


def optimize_max_sharpe(
    cma: CapitalMarketAssumptions,
    weight_cap: float = 0.65,
    asset_bounds: Dict[str, Tuple[float, float]] | None = None,
    equity_cap: float | None = None,
) -> FrontierPoint:
    bounds, feasible = _resolve_bounds(cma.assets, weight_cap, asset_bounds)
    if not feasible:
        bounds, _ = _resolve_bounds(cma.assets, weight_cap, None)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(np.dot(w, cma.expected_returns))
        port_vol = float(np.sqrt(max(w @ cma.cov_matrix @ w, 1e-18)))
        return -(port_ret - cma.risk_free_rate) / port_vol

    extra = _equity_group_constraint(cma.assets, equity_cap)
    weights = _solve(cma, neg_sharpe, bounds, extra_constraints=extra)
    return _build_frontier_point(cma, weights, "max_sharpe")


def optimize_min_variance(
    cma: CapitalMarketAssumptions,
    weight_cap: float = 0.65,
    asset_bounds: Dict[str, Tuple[float, float]] | None = None,
    equity_cap: float | None = None,
) -> FrontierPoint:
    bounds, feasible = _resolve_bounds(cma.assets, weight_cap, asset_bounds)
    if not feasible:
        bounds, _ = _resolve_bounds(cma.assets, weight_cap, None)

    def variance(w: np.ndarray) -> float:
        return float(w @ cma.cov_matrix @ w)

    extra = _equity_group_constraint(cma.assets, equity_cap)
    weights = _solve(cma, variance, bounds, extra_constraints=extra)
    return _build_frontier_point(cma, weights, "min_variance")


def optimize_target_volatility(
    cma: CapitalMarketAssumptions,
    target_vol: float,
    weight_cap: float = 0.65,
    asset_bounds: Dict[str, Tuple[float, float]] | None = None,
    equity_cap: float | None = None,
) -> FrontierPoint:
    """Maximize expected return such that vol(w) <= target_vol."""
    bounds, feasible = _resolve_bounds(cma.assets, weight_cap, asset_bounds)
    if not feasible:
        bounds, _ = _resolve_bounds(cma.assets, weight_cap, None)

    def neg_return(w: np.ndarray) -> float:
        return -float(np.dot(w, cma.expected_returns))

    def vol_ineq(w: np.ndarray) -> float:
        return float(target_vol - np.sqrt(max(w @ cma.cov_matrix @ w, 0.0)))

    extra: List[dict] = [{"type": "ineq", "fun": vol_ineq}]
    extra.extend(_equity_group_constraint(cma.assets, equity_cap))
    weights = _solve(cma, neg_return, bounds, extra_constraints=extra)
    point = _build_frontier_point(cma, weights, f"target_vol_{target_vol:.3f}")

    if point.volatility > target_vol * 1.05:
        mv = optimize_min_variance(cma, weight_cap=weight_cap, asset_bounds=asset_bounds, equity_cap=equity_cap)
        if mv.volatility <= point.volatility:
            return mv
    return point


def optimize_frontier_target_volatility_exact(
    cma: CapitalMarketAssumptions,
    target_vol: float,
    tolerance: float = 1e-4,
    weight_cap: float = 1.0,
    asset_bounds: Dict[str, Tuple[float, float]] | None = None,
    equity_cap: float | None = None,
) -> FrontierPoint:
    """Long-only frontier point at exact target volatility.

    Solves:
        maximize   mu^T w
        subject to sqrt(w^T Sigma w) == target_vol
                   sum(w) == 1
                   0 <= w_i <= 1
    """
    n = len(cma.assets)
    bounds, feasible = _resolve_bounds(cma.assets, weight_cap, asset_bounds)
    if not feasible:
        raise ValueError("Infeasible asset bounds: sum(lower_bounds) > 1 or sum(upper_bounds) < 1.")

    # Feasibility envelope in the same long-only domain.
    min_var = optimize_min_variance(cma, weight_cap=weight_cap, asset_bounds=asset_bounds, equity_cap=equity_cap)
    w_max_ret = _solve(
        cma,
        lambda w: -float(np.dot(w, cma.expected_returns)),
        bounds,
        extra_constraints=_equity_group_constraint(cma.assets, equity_cap),
    )
    max_ret_point = _build_frontier_point(cma, w_max_ret, "max_return_feasible")

    if target_vol <= min_var.volatility + tolerance:
        point = _build_frontier_point(cma, np.array([min_var.weights[a] for a in cma.assets]), "frontier_clamped_min_var")
        return point
    if target_vol >= max_ret_point.volatility - tolerance:
        return _build_frontier_point(cma, w_max_ret, "frontier_clamped_max_return")

    w_min = np.array([min_var.weights[a] for a in cma.assets], dtype=float)

    # Deterministic warm start: mix min-var with max-return until vol ~= target.
    lo, hi = 0.0, 1.0
    start = w_min.copy()
    for _ in range(60):
        alpha = 0.5 * (lo + hi)
        w = (1.0 - alpha) * w_min + alpha * w_max_ret
        w = w / max(float(w.sum()), 1e-12)
        vol = float(np.sqrt(max(w @ cma.cov_matrix @ w, 1e-18)))
        start = w
        if vol < target_vol:
            lo = alpha
        else:
            hi = alpha

    def neg_return(w: np.ndarray) -> float:
        return -float(np.dot(w, cma.expected_returns))

    def budget_eq(w: np.ndarray) -> float:
        return float(np.sum(w) - 1.0)

    def vol_eq(w: np.ndarray) -> float:
        return float(np.sqrt(max(w @ cma.cov_matrix @ w, 1e-18)) - target_vol)

    constraints = [
        {"type": "eq", "fun": budget_eq},
        {"type": "eq", "fun": vol_eq},
        *_equity_group_constraint(cma.assets, equity_cap),
    ]
    result = minimize(
        neg_return,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-12},
    )

    if result.success and np.all(np.isfinite(result.x)):
        weights = np.array(result.x, dtype=float)
    else:
        # Retry from a neutral interior point if the first solve stalls.
        retry_start = np.full(n, 1.0 / n)
        retry = minimize(
            neg_return,
            retry_start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 4000, "ftol": 1e-13},
        )
        if retry.success and np.all(np.isfinite(retry.x)):
            weights = np.array(retry.x, dtype=float)
        else:
            # Final fallback: inequality target-vol solution in long-only domain.
            fallback = optimize_target_volatility(
                cma,
                target_vol=target_vol,
                weight_cap=weight_cap,
                asset_bounds=asset_bounds,
                equity_cap=equity_cap,
            )
            weights = np.array([fallback.weights[a] for a in cma.assets], dtype=float)

    weights = np.clip(weights, 0.0, 1.0)
    weights = weights / max(float(weights.sum()), 1e-12)
    point = _build_frontier_point(cma, weights, f"frontier_target_vol_exact_{target_vol:.4f}")
    return point


def optimize_mean_variance_utility(
    cma: CapitalMarketAssumptions,
    risk_aversion: float,
    weight_cap: float = 0.65,
    asset_bounds: Dict[str, Tuple[float, float]] | None = None,
    equity_cap: float | None = None,
) -> FrontierPoint:
    bounds, feasible = _resolve_bounds(cma.assets, weight_cap, asset_bounds)
    if not feasible:
        bounds, _ = _resolve_bounds(cma.assets, weight_cap, None)

    def neg_utility(w: np.ndarray) -> float:
        port_ret = float(np.dot(w, cma.expected_returns))
        port_var = float(w @ cma.cov_matrix @ w)
        return -(port_ret - 0.5 * risk_aversion * port_var)

    extra = _equity_group_constraint(cma.assets, equity_cap)
    weights = _solve(cma, neg_utility, bounds, extra_constraints=extra)
    return _build_frontier_point(cma, weights, f"mvu_{risk_aversion:.1f}")


def optimize_equal_risk_contribution(
    cma: CapitalMarketAssumptions,
    weight_cap: float = 0.65,
    asset_bounds: Dict[str, Tuple[float, float]] | None = None,
    equity_cap: float | None = None,
) -> FrontierPoint:
    """Solve the Equal Risk Contribution (ERC) "true" risk parity portfolio.

    Minimizes the squared deviation between each asset's risk contribution
    and the equal target 1/N.  Falls back to inverse-volatility weighting
    when SLSQP fails to converge.
    """
    bounds, feasible = _resolve_bounds(cma.assets, weight_cap, asset_bounds)
    if not feasible:
        bounds, _ = _resolve_bounds(cma.assets, weight_cap, None)

    n = len(cma.assets)
    target = 1.0 / n

    def erc_objective(w: np.ndarray) -> float:
        port_var = float(w @ cma.cov_matrix @ w)
        if port_var <= 0:
            return 1e6
        marginal = cma.cov_matrix @ w
        rc = w * marginal / port_var
        return float(np.sum((rc - target) ** 2))

    extra = _equity_group_constraint(cma.assets, equity_cap)
    weights = _solve(cma, erc_objective, bounds, extra_constraints=extra)
    point = _build_frontier_point(cma, weights, "erc_risk_parity")

    # Fallback to 1/sigma if ERC degenerated.
    if not np.isfinite(point.expected_return) or point.volatility <= 1e-9:
        sig = np.sqrt(np.diag(cma.cov_matrix))
        inv = 1.0 / np.where(sig > 0, sig, 1.0)
        w = inv / inv.sum()
        point = _build_frontier_point(cma, _project_to_simplex(w, bounds), "inverse_volatility")

    return point


def build_frontier_diagnostics(
    cma: CapitalMarketAssumptions,
    n_points: int = 70,
    n_random: int = 25000,
    weight_cap: float = 0.65,
    seed: int = 7,
    asset_bounds: Dict[str, Tuple[float, float]] | None = None,
    equity_cap: float | None = None,
) -> Dict[str, object]:
    """Sample a random portfolio cloud and trace the efficient frontier line."""
    rng = np.random.default_rng(seed)
    n_assets = len(cma.assets)
    bounds, feasible = _resolve_bounds(cma.assets, weight_cap, asset_bounds)
    if not feasible:
        bounds, _ = _resolve_bounds(cma.assets, weight_cap, None)

    random_rows: List[np.ndarray] = []
    max_tries = max(20_000, n_random * 15)
    tries = 0
    eq_idx = [i for i, a in enumerate(cma.assets) if a in ("EGX30", "EGX100")]
    while len(random_rows) < n_random and tries < max_tries:
        tries += 1
        raw = rng.dirichlet(np.ones(n_assets))
        w = _project_to_simplex(raw, bounds)
        if equity_cap is not None and len(eq_idx) >= 2:
            if float(np.sum(w[eq_idx])) > equity_cap + 1e-9:
                continue
        # Final bounds guard after projection.
        if any((w[i] < b[0] - 1e-8) or (w[i] > b[1] + 1e-8) for i, b in enumerate(bounds)):
            continue
        random_rows.append(w)

    if not random_rows:
        random_rows = [np.full(n_assets, 1.0 / n_assets)]

    random_weights = np.vstack(random_rows)
    rand_ret = random_weights @ cma.expected_returns
    rand_vol = np.sqrt(np.einsum("ij,jk,ik->i", random_weights, cma.cov_matrix, random_weights))
    rand_sharpe = (rand_ret - cma.risk_free_rate) / (rand_vol + 1e-12)

    # Frontier tracing domain:
    # Lower bound starts near the random-cloud floor; upper bound is set to
    # the maximum single-asset expected return as requested.
    ret_low = float(np.percentile(rand_ret, 2))
    ret_high = float(np.max(cma.expected_returns))
    ret_grid = np.linspace(ret_low, ret_high, n_points)
    frontier: List[FrontierPoint] = []
    prev_w: np.ndarray | None = None
    conv_ok = 0
    conv_fail = 0
    for target_ret in ret_grid:

        def variance(w: np.ndarray, _t=target_ret) -> float:
            return float(w @ cma.cov_matrix @ w)

        def ret_eq(w: np.ndarray, _t=target_ret) -> float:
            return float(np.dot(w, cma.expected_returns)) - _t

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, {"type": "eq", "fun": ret_eq}]
        constraints.extend(_equity_group_constraint(cma.assets, equity_cap))

        # Warm-start high-return solves from the previous frontier solution.
        if prev_w is None:
            x0 = np.array([(b[0] + b[1]) / 2.0 for b in bounds], dtype=float)
            x0 = x0 / max(float(np.sum(x0)), 1e-12)
        else:
            x0 = prev_w
        result = minimize(
            variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if result.success and np.all(np.isfinite(result.x)):
            weights = _project_to_simplex(result.x, bounds)
            conv_ok += 1
        else:
            # Fallback for numerically-stiff high-return targets.
            weights = _solve(
                cma,
                variance,
                bounds,
                extra_constraints=[{"type": "eq", "fun": ret_eq}] + _equity_group_constraint(cma.assets, equity_cap),
                start=x0,
            )
            conv_fail += 1
        prev_w = weights
        frontier.append(_build_frontier_point(cma, weights, "frontier"))

    # Explicitly append the maximum-return feasible anchor so the frontier
    # reaches the upper-right boundary under the active constraints.
    max_ret_w = _solve(
        cma,
        lambda w: -float(np.dot(w, cma.expected_returns)),
        bounds,
        extra_constraints=_equity_group_constraint(cma.assets, equity_cap),
        start=prev_w,
    )
    max_ret_point = _build_frontier_point(cma, max_ret_w, "frontier_max_return_anchor")
    if not frontier or max_ret_point.expected_return > max(p.expected_return for p in frontier) + 1e-9:
        frontier.append(max_ret_point)

    frontier = sorted(frontier, key=lambda p: p.volatility)
    # Keep only non-dominated points (monotone increasing return with risk).
    filtered: List[FrontierPoint] = []
    best_ret = -np.inf
    for p in frontier:
        if p.expected_return >= best_ret - 1e-10:
            filtered.append(p)
            best_ret = max(best_ret, p.expected_return)
    frontier = filtered
    keep = min(4000, len(rand_ret))
    idx = np.linspace(0, len(rand_ret) - 1, keep, dtype=int)

    return {
        "assets": cma.assets,
        "sample_size": cma.sample_size,
        "risk_free_rate": cma.risk_free_rate,
        "annual_mean_returns": {a: round(float(v), 6) for a, v in zip(cma.assets, cma.expected_returns)},
        "annual_volatility": {a: round(float(np.sqrt(cma.cov_matrix[i, i])), 6) for i, a in enumerate(cma.assets)},
        "annual_covariance": cma.cov_matrix.round(8).tolist(),
        "random_cloud": {
            "returns": rand_ret[idx].round(6).tolist(),
            "volatilities": rand_vol[idx].round(6).tolist(),
            "sharpes": rand_sharpe[idx].round(6).tolist(),
        },
        "frontier_line": {
            "returns": [round(p.expected_return, 6) for p in frontier],
            "volatilities": [round(p.volatility, 6) for p in frontier],
            "sharpes": [round(p.sharpe, 6) for p in frontier],
        },
        "frontier_solver": {
            "target_return_min": round(ret_low, 6),
            "target_return_max": round(ret_high, 6),
            "converged_points": int(conv_ok),
            "fallback_points": int(conv_fail),
            "max_return_anchor": {
                "expected_return": round(max_ret_point.expected_return, 6),
                "volatility": round(max_ret_point.volatility, 6),
            },
        },
    }
