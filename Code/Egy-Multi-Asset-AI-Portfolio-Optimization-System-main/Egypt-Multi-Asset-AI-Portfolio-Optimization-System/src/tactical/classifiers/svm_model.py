from __future__ import annotations

from sklearn.svm import SVC


def build_svm_model() -> SVC:
    return SVC(kernel="rbf", C=1.2, probability=True, class_weight="balanced")
