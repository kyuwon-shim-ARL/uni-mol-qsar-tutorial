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


def per_fold_top_features(
    fold_models: list,
    X: np.ndarray,
    fold_indices: list,
    feature_names: list[str],
    k: int = 20,
    explain_on: str = "train",
) -> list[list[tuple[str, float]]]:
    """For each fold, fit-time-locked TreeSHAP top-k features.

    `explain_on` selects which rows go into SHAP: "train" (default — what
    the model saw) or "val" (what was held out). Both are valid; train is
    less noisy at small fold sizes.
    """
    out: list[list[tuple[str, float]]] = []
    for model, (tr, va) in zip(fold_models, fold_indices):
        idx = tr if explain_on == "train" else va
        res = explain(model, X[idx], feature_names=feature_names)
        out.append(res.top_features(k=k))
    return out


def stability_summary(per_fold_top: list[list[tuple[str, float]]]) -> list[dict]:
    """Aggregate per-fold top-k lists into a stability table.

    Returns rows sorted by fold_presence DESC then mean_rank ASC:
        {feature, fold_presence, fold_presence_ratio, mean_rank, rank_std,
         mean_importance}
    No threshold is applied. Distribution is preserved so the caller can
    decide what is "stable enough" (5/5 = strong, 1-2/5 = likely spurious).
    """
    from collections import defaultdict

    n_folds = len(per_fold_top)
    counts: dict[str, int] = defaultdict(int)
    ranks: dict[str, list[int]] = defaultdict(list)
    imps: dict[str, list[float]] = defaultdict(list)
    for fold_top in per_fold_top:
        for rank, (feat, imp) in enumerate(fold_top, start=1):
            counts[feat] += 1
            ranks[feat].append(rank)
            imps[feat].append(float(imp))

    rows: list[dict] = []
    for feat, c in counts.items():
        rks = ranks[feat]
        rows.append({
            "feature": feat,
            "fold_presence": int(c),
            "fold_presence_ratio": float(c / n_folds),
            "mean_rank": float(np.mean(rks)),
            "rank_std": float(np.std(rks)) if len(rks) > 1 else 0.0,
            "mean_importance": float(np.mean(imps[feat])),
        })
    rows.sort(key=lambda r: (-r["fold_presence"], r["mean_rank"]))
    return rows
