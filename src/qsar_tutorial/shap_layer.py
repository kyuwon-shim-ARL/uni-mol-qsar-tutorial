"""TreeSHAP wrapper.

Why TreeSHAP and not KernelSHAP: tree-model SHAP is exact in polynomial time;
KernelSHAP is approximate and slow. Always use TreeSHAP when the underlying
model is a tree ensemble (XGBoost / LightGBM / RandomForest).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShapResult:
    values: np.ndarray            # (N, D) per-sample, per-feature contributions to P(active)
    base_value: float             # expected model output
    feature_names: list[str]

    def global_importance(self) -> np.ndarray:
        """Mean absolute SHAP per feature — the standard global importance."""
        return np.mean(np.abs(self.values), axis=0)

    def top_features(self, k: int = 20) -> list[tuple[str, float]]:
        imp = self.global_importance()
        idx = np.argsort(-imp)[:k]
        return [(self.feature_names[i], float(imp[i])) for i in idx]


def explain(model, X: np.ndarray, feature_names: list[str] | None = None) -> ShapResult:
    """Run TreeSHAP. For binary XGBoost, returns contributions to log-odds of class=1."""
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    # Older shap returns list per class; newer returns ndarray for binary.
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (N, D, n_classes) for some xgb wrappers
        sv = sv[..., 1]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])
    names = feature_names or [f"f{i}" for i in range(sv.shape[1])]
    return ShapResult(values=sv, base_value=float(base), feature_names=names)
