"""Risk-parity weighting helpers.

Two flavors are exposed:

* :func:`inverse_volatility_weights` -- naive 1/sigma weights, useful as a
  reference point and as a fallback when ERC fails to converge.
* :func:`equal_risk_contribution` -- delegates to the institutional ERC
  solver in :mod:`efficient_frontier` so risk-parity output is consistent
  with the rest of the optimization stack.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.strategic.optimization.efficient_frontier import (
    CapitalMarketAssumptions,
    optimize_equal_risk_contribution,
)


def inverse_volatility_weights(returns: pd.DataFrame) -> Dict[str, float]:
    vol = returns.std().replace(0, np.nan).dropna()
    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()
    return {asset: float(w) for asset, w in weights.items()}


def equal_risk_contribution(cma: CapitalMarketAssumptions) -> Dict[str, float]:
    point = optimize_equal_risk_contribution(cma)
    return point.weights
