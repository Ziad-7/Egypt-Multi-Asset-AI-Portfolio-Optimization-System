"""EM-based estimator for multivariate-Gaussian return panels with missing data.

The legacy loader required ``dropna(how="any")`` over the active universe
which truncated the panel to ~180 days because Gold has only existed
since 2024-06.  This module replaces that path with the canonical
Dempster-Laird-Rubin (1977) / Schafer (1997 Section 5) EM estimator for
a multivariate-normal model with arbitrary missing-at-random patterns.

The estimator iterates between

    E-step: for each row i partition the columns into observed (O) and
            missing (M) and impute the conditional mean and conditional
            covariance under the current (mu, Sigma).
    M-step: update mu from the mean of the completed rows; update Sigma
            from the sum of the squared deviations *plus* the average
            of the conditional residual covariances (the "T2" term).

A nearest-PSD projection is applied to Sigma after each iteration so
the solution is always usable by the optimizer.  When the conditional
``Sigma_OO`` block is near-singular for some row pattern, we shrink
that block toward its diagonal before inverting -- a per-row analogue
of the diagonal-preserving correlation shrinkage used elsewhere in the
platform.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class EMResult:
    """Output bundle for downstream consumers."""

    mu: np.ndarray
    cov: np.ndarray
    completed: pd.DataFrame
    iterations: int
    converged: bool
    delta_history: List[float]
    imputed_fraction: Dict[str, float]
    imputation_residual_var: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "final_delta": float(self.delta_history[-1]) if self.delta_history else None,
            "imputed_fraction": {k: round(float(v), 4) for k, v in self.imputed_fraction.items()},
            "imputation_residual_var": {k: round(float(v), 8) for k, v in self.imputation_residual_var.items()},
        }


def _nearest_psd(matrix: np.ndarray, eigen_floor_ratio: float = 1e-10) -> np.ndarray:
    """Symmetrize and clip negative eigenvalues -- a Higham 2002 lite."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    floor = max(eigen_floor_ratio * float(np.trace(sym)), 1e-12)
    eigvals = np.clip(eigvals, floor, None)
    return (eigvecs * eigvals) @ eigvecs.T


def _pairwise_cov(returns: pd.DataFrame) -> np.ndarray:
    """Bias-corrected pairwise-deletion covariance, PSD-projected.

    Used as the EM warm start.  pandas ``cov(min_periods=2)`` gives us
    an estimator that uses overlapping observations cell-by-cell; we
    then project to nearest PSD so subsequent linear-algebra is stable.
    """
    cov = returns.cov(min_periods=2).values
    cov = np.where(np.isfinite(cov), cov, 0.0)
    return _nearest_psd(cov)


def _safe_inverse(matrix: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Invert with a tiny Tikhonov ridge fallback if the matrix is singular."""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.inv(matrix + np.eye(matrix.shape[0]) * ridge)


def em_impute_returns(
    returns: pd.DataFrame,
    max_iter: int = 150,
    tol: float = 1e-7,
    min_observed_rows: int = 5,
    return_completed: bool = True,
) -> EMResult:
    """Estimate (mu, Sigma) and a completed return panel via EM.

    Parameters
    ----------
    returns
        Wide-format daily-return panel with NaNs in the cells that are
        missing for a given asset on a given date.
    max_iter, tol
        Standard EM controls.  ``tol`` is measured against the relative
        Frobenius-norm change in Sigma.
    min_observed_rows
        Each column must have at least this many observed rows or it is
        excluded from the EM model.
    """
    if returns.empty:
        raise ValueError("EM imputer received an empty frame.")

    cols = list(returns.columns)
    counts = returns.notna().sum()
    keep = [c for c in cols if counts[c] >= min_observed_rows]
    dropped = [c for c in cols if c not in keep]

    R = returns[keep].values.astype(float).copy()
    n_rows, n_cols = R.shape
    if n_cols == 0:
        raise ValueError("No columns retained after min_observed_rows filter.")

    # Warm start.
    col_mean = np.array([np.nanmean(R[:, j]) if counts[keep[j]] else 0.0 for j in range(n_cols)])
    mu = np.where(np.isfinite(col_mean), col_mean, 0.0)
    sigma = _pairwise_cov(returns[keep])
    sigma = _nearest_psd(sigma)

    is_observed = ~np.isnan(R)
    completed = np.where(is_observed, R, np.tile(mu, (n_rows, 1)))

    delta_history: List[float] = []
    converged = False

    for iteration in range(1, max_iter + 1):
        # Accumulators for the M-step covariance update.
        sum_xx = np.zeros((n_cols, n_cols))
        sum_x = np.zeros(n_cols)
        residual_cov_sum = np.zeros((n_cols, n_cols))

        for i in range(n_rows):
            obs = is_observed[i]
            if obs.all():
                completed[i] = R[i]
            else:
                miss = ~obs
                # Edge case: no observed columns -> use unconditional mean.
                if not obs.any():
                    completed[i, miss] = mu[miss]
                    residual_cov_block = sigma[np.ix_(miss, miss)]
                else:
                    sigma_oo = sigma[np.ix_(obs, obs)]
                    sigma_mo = sigma[np.ix_(miss, obs)]
                    sigma_mm = sigma[np.ix_(miss, miss)]
                    # Diagonal shrinkage if near-singular.
                    diag_oo = np.diag(np.diag(sigma_oo))
                    lam = 0.0
                    cond = np.linalg.cond(sigma_oo) if sigma_oo.size else 1.0
                    if not np.isfinite(cond) or cond > 1e10:
                        lam = 0.20
                    sigma_oo_use = (1.0 - lam) * sigma_oo + lam * diag_oo
                    inv_oo = _safe_inverse(sigma_oo_use)
                    deviation = R[i, obs] - mu[obs]
                    cond_mean = mu[miss] + sigma_mo @ inv_oo @ deviation
                    completed[i, miss] = cond_mean
                    residual_cov_block = sigma_mm - sigma_mo @ inv_oo @ sigma_mo.T

                # Embed the residual cov for missing cells back into a full p x p block.
                full_residual = np.zeros((n_cols, n_cols))
                m_idx = np.where(miss)[0]
                for a, mi in enumerate(m_idx):
                    for b, mj in enumerate(m_idx):
                        full_residual[mi, mj] = residual_cov_block[a, b]
                residual_cov_sum += full_residual

            sum_x += completed[i]
            sum_xx += np.outer(completed[i], completed[i])

        mu_new = sum_x / n_rows
        sigma_new = sum_xx / n_rows - np.outer(mu_new, mu_new) + residual_cov_sum / n_rows
        sigma_new = _nearest_psd(sigma_new)

        denom = np.linalg.norm(sigma) + 1e-12
        delta = np.linalg.norm(sigma_new - sigma) / denom
        delta_history.append(float(delta))

        mu, sigma = mu_new, sigma_new
        if delta < tol:
            converged = True
            break

    iterations = iteration

    completed_df = pd.DataFrame(completed, index=returns.index, columns=keep)
    # Restore dropped columns as NaN-filled to preserve schema.
    for c in dropped:
        completed_df[c] = np.nan
    completed_df = completed_df[cols]

    imputed_fraction = {
        c: float((returns[c].isna() & completed_df[c].notna()).sum()) / max(len(returns), 1) for c in cols
    }
    # Per-asset diagnostic: variance of imputed values vs variance of observed values.
    residual_variance: Dict[str, float] = {}
    for j, c in enumerate(keep):
        if returns[c].isna().any():
            mask = returns[c].isna().values
            imputed_vals = completed[mask, j]
            residual_variance[c] = float(np.var(imputed_vals)) if imputed_vals.size > 1 else 0.0
        else:
            residual_variance[c] = 0.0

    if not return_completed:
        completed_df = pd.DataFrame()

    # Re-pad mu / sigma to original column ordering, NaN for dropped cols.
    mu_full = np.full(len(cols), np.nan)
    sigma_full = np.full((len(cols), len(cols)), np.nan)
    for i, c in enumerate(cols):
        if c in keep:
            j = keep.index(c)
            mu_full[i] = mu[j]
            for k, d in enumerate(cols):
                if d in keep:
                    l = keep.index(d)
                    sigma_full[i, k] = sigma[j, l]

    return EMResult(
        mu=mu_full,
        cov=sigma_full,
        completed=completed_df,
        iterations=int(iterations),
        converged=bool(converged),
        delta_history=delta_history,
        imputed_fraction=imputed_fraction,
        imputation_residual_var=residual_variance,
    )
