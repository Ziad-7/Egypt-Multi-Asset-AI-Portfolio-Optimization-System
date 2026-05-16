from __future__ import annotations

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def build_xgboost_model(n_classes: int):
    if XGBClassifier is None:
        return None
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=42,
        verbosity=0,
    )
