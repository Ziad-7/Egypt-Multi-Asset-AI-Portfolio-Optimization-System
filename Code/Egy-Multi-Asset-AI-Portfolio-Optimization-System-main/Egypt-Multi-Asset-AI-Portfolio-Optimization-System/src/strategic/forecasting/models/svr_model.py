from __future__ import annotations

import numpy as np
from sklearn.svm import SVR

from .common import ModelRunResult, walk_forward_model_eval


def run_svr(X: np.ndarray, y: np.ndarray) -> ModelRunResult:
    return walk_forward_model_eval(
        X=X,
        y=y,
        model_name="svr",
        model_factory=lambda: SVR(C=1.0, epsilon=1e-3, kernel="rbf"),
    )
