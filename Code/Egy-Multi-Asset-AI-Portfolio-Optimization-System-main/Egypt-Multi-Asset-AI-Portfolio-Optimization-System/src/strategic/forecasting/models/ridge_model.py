from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .common import ModelRunResult, walk_forward_model_eval


def run_ridge(X: np.ndarray, y: np.ndarray) -> ModelRunResult:
    return walk_forward_model_eval(
        X=X,
        y=y,
        model_name="ridge",
        model_factory=lambda: Ridge(alpha=1.0),
    )
