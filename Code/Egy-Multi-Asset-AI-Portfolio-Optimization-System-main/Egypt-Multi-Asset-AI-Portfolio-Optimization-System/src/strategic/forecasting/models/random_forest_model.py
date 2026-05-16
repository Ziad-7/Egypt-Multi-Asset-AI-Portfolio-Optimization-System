from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .common import ModelRunResult, walk_forward_model_eval


def run_random_forest(X: np.ndarray, y: np.ndarray) -> ModelRunResult:
    return walk_forward_model_eval(
        X=X,
        y=y,
        model_name="random_forest",
        model_factory=lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42,
        ),
    )
