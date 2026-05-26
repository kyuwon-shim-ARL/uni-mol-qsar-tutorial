"""XGBoost binary classifier on top of any feature matrix.

Deliberately wraps XGBClassifier with a small surface area so TreeSHAP works
out of the box. Use stratified K-fold OOF for honest evaluation — the parent
project's headline lesson is that leakage between feature extraction and
fold splitting inflates AUC by ~0.13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


DEFAULT_PARAMS: dict[str, Any] = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)


@dataclass
class OOFResult:
    y_true: np.ndarray
    y_prob: np.ndarray              # P(active) per sample, from out-of-fold predictions
    per_fold: list[dict[str, float]] = field(default_factory=list)

    @property
    def auc(self) -> float:
        return float(roc_auc_score(self.y_true, self.y_prob))

    @property
    def auprc(self) -> float:
        return float(average_precision_score(self.y_true, self.y_prob))

    @property
    def mcc(self) -> float:
        return float(matthews_corrcoef(self.y_true, (self.y_prob >= 0.5).astype(int)))

    def summary(self) -> dict[str, float]:
        return {"AUC": self.auc, "AUPRC": self.auprc, "MCC@0.5": self.mcc}


def build_classifier(class_weight: str | None = "balanced", **overrides):
    """Return a fresh XGBClassifier with default + override params."""
    import xgboost as xgb

    params = {**DEFAULT_PARAMS, **overrides}
    return xgb.XGBClassifier(**params)


def fit_with_class_weight(model, X: np.ndarray, y: np.ndarray) -> None:
    """Balanced class-weight fit — antibacterial datasets are usually skewed."""
    sw = compute_sample_weight("balanced", y)
    model.fit(X, y, sample_weight=sw)


def cross_validated_oof(
    X: np.ndarray,
    y: np.ndarray,
    folds,
    **params,
) -> OOFResult:
    """Stratified K-fold OOF probability for class=1 (active)."""
    y_prob = np.zeros(len(y), dtype=float)
    per_fold = []
    for k, (tr, va) in enumerate(folds, start=1):
        model = build_classifier(**params)
        fit_with_class_weight(model, X[tr], y[tr])
        proba = model.predict_proba(X[va])[:, 1]
        y_prob[va] = proba
        try:
            fold_auc = roc_auc_score(y[va], proba)
        except ValueError:
            fold_auc = float("nan")  # fold lacks a class
        per_fold.append({"fold": k, "n_val": len(va), "AUC": float(fold_auc)})
    return OOFResult(y_true=y.copy(), y_prob=y_prob, per_fold=per_fold)
