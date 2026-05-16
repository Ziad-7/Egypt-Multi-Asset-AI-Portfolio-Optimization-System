from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_logistic_model() -> LogisticRegression:
    return LogisticRegression(max_iter=1200, class_weight="balanced")
