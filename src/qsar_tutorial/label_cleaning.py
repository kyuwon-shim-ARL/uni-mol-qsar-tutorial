"""Active-class-protected Cleanlab label cleaning for binary QSAR.

This is the binary-classification analogue of the parent project's
P-protected 4-class Cleanlab cleaner. The minority class (active=1) is
small enough that Cleanlab's "flag and relabel to argmax" behavior would
otherwise wash most of it into the majority (inactive=0). We follow the
same recipe used in the parent project's `run_finetune.py`:

    1. Stratified-K-fold OOF probabilities from the supplied features
       (whatever the pipeline is using — ECFP4 fingerprint, Uni-Mol
       embedding, etc.).
    2. `cleanlab.filter.find_label_issues` flags suspicious rows.
    3. Flagged rows with original label 0 (inactive) → relabel to
       argmax of OOF prob.
    4. Flagged rows with original label 1 (active) → **leave untouched**.
       The minority class is protected by construction.

Why protect active (not inactive):
    - Active is ~2.86% (689/24,120). The classifier baseline is bad at
      it; many active rows look like "errors" to Cleanlab.
    - Active is also the design target — losing it makes downstream
      MMP / SHAP / SAE interpretation meaningless.
    - This mirrors the parent project's P-protected logic exactly,
      adapted to binary.

Why offer it at all:
    - The censored-MIC issue documented in data/README.md inflates the
      active class with false positives. Cleanlab gives a second,
      *model-agreement*-based view of which labels look noisy.
    - The expected behaviour: Cleanlab will relabel some "inactive"
      rows to active (rescuing missed positives) but cannot remove
      false-positive actives because they are protected. The cleaned
      label set is therefore conservative — useful for sensitivity
      analysis, not a replacement for the strict-MIC label.

This module is intentionally optional. The pipeline's default behavior
is unchanged; pass `--cleanlab` in `examples/run_full_pipeline.py` to
turn it on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class CleanlabReport:
    """Audit trail for a single cleanlab pass.

    All counts are integers; the cleaned label array is returned alongside
    this report so the caller can pass it to downstream model training.
    """
    n_total: int
    n_flagged: int
    n_relabeled: int
    n_protected: int            # flagged but active=1 → kept
    n_changed: int              # final relabel count (== n_relabeled here)
    active_before: int
    active_after: int
    pct_changed: float
    per_fold_auc: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_flagged": self.n_flagged,
            "n_relabeled": self.n_relabeled,
            "n_protected_active": self.n_protected,
            "n_changed": self.n_changed,
            "pct_changed": round(self.pct_changed, 3),
            "active_before": self.active_before,
            "active_after": self.active_after,
            "per_fold_auc_mean": (
                round(float(np.mean(self.per_fold_auc)), 4)
                if self.per_fold_auc else None
            ),
        }


def _oof_predict_probs(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    **xgb_overrides: Any,
) -> tuple[np.ndarray, list[float]]:
    """5-fold stratified OOF binary probabilities via XGBoost.

    Uses the same classifier the pipeline uses downstream (see model.py),
    not a separate model. Cleanlab needs class probabilities, not
    decisions; we provide both columns (P(0), P(1)).
    """
    from sklearn.metrics import roc_auc_score

    from . import model as model_mod

    n = len(y)
    probs = np.zeros((n, 2), dtype=np.float64)
    per_fold_auc: list[float] = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, va in skf.split(np.zeros(n), y):
        clf = model_mod.build_classifier(**xgb_overrides)
        sw = compute_sample_weight("balanced", y[tr])
        clf.fit(X[tr], y[tr], sample_weight=sw)
        p1 = clf.predict_proba(X[va])[:, 1]
        probs[va, 1] = p1
        probs[va, 0] = 1.0 - p1
        try:
            per_fold_auc.append(float(roc_auc_score(y[va], p1)))
        except ValueError:
            per_fold_auc.append(float("nan"))
    return probs, per_fold_auc


def clean_labels_active_protected(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    **xgb_overrides: Any,
) -> tuple[np.ndarray, CleanlabReport]:
    """Run active-protected Cleanlab cleaning, return (y_cleaned, report).

    Args:
        X:  feature matrix the pipeline is already using (shape (n, d)).
        y:  binary labels (0 = inactive, 1 = active).
        n_splits, random_state:  stratified K-fold config for OOF probs.
        xgb_overrides:  passed through to `model.build_classifier`.

    Returns:
        y_cleaned:   array of same shape as `y`, with some 0s flipped to 1.
                     No 1 is ever flipped to 0 (active class is protected).
        report:      `CleanlabReport` with audit numbers.
    """
    try:
        from cleanlab.filter import find_label_issues
    except ImportError as e:
        raise RuntimeError(
            "cleanlab is required for label cleaning. "
            "Add cleanlab>=2.6 to your environment."
        ) from e

    y = np.asarray(y, dtype=int)
    n = len(y)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError(f"clean_labels_active_protected: labels must be in {{0,1}}, got {set(y.tolist())}")

    probs, per_fold_auc = _oof_predict_probs(
        X, y, n_splits=n_splits, random_state=random_state, **xgb_overrides
    )
    issue_mask = find_label_issues(labels=y, pred_probs=probs)
    n_flagged = int(issue_mask.sum())

    y_cleaned = y.copy()
    n_relabeled = 0
    n_protected = 0
    for i in np.where(issue_mask)[0]:
        if y[i] == 1:
            n_protected += 1
            continue
        new_label = int(np.argmax(probs[i]))
        if new_label != y[i]:
            y_cleaned[i] = new_label
            n_relabeled += 1

    # Protect-class invariant: no active label was changed.
    assert (y[y == 1] == y_cleaned[y == 1]).all(), \
        "active labels were modified — protection invariant violated"

    n_changed = int((y != y_cleaned).sum())
    report = CleanlabReport(
        n_total=n,
        n_flagged=n_flagged,
        n_relabeled=n_relabeled,
        n_protected=n_protected,
        n_changed=n_changed,
        active_before=int(y.sum()),
        active_after=int(y_cleaned.sum()),
        pct_changed=100.0 * n_changed / max(n, 1),
        per_fold_auc=per_fold_auc,
    )
    return y_cleaned, report
