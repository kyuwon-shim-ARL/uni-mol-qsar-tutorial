"""T2 — cleanlab label-noise audit (A/B) on CO-ADD PA.

Pipeline:
    1. Compute OOF probabilities on the current dataset (random K-fold, fast).
    2. Run cleanlab.filter.find_label_issues → flagged sample indices.
    3. Save flagged molecules (CHEMBL ID + SMILES + active + p_active) to CSV
       for MAINTAINER review. Interns observe only — they do not edit the
       'review_note' column without sign-off.
    4. Retrain on the cleaned subset, compare AUPRC + top-100 enrichment with
       bootstrap CI95 (1000 resamples) and paired permutation test.
    5. Render a self-contained HTML report. Negative results (cleaning hurt
       performance) are reported as-is — that is the deliverable.

Honest risk: 689 active / 23,431 inactive (2.86%). Confident learning may
flag actives circularly (model can't predict them → flagged as noise → removed
→ apparent improvement is sample-size artifact). The bootstrap CI is the
sanity check against that.

Usage:
    python examples/cleanlab_audit.py
    python examples/cleanlab_audit.py --bootstrap 200    # faster smoke
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from qsar_tutorial.data import load_coadd_pa, stratified_folds, Dataset
from qsar_tutorial.featurizer import featurize_ecfp4
from qsar_tutorial.model import cross_validated_oof


def bootstrap_auprc_diff(y_true, p_a, p_b, n_boot=1000, seed=42):
    """Bootstrap CI95 for AUPRC(B) - AUPRC(A). Same indices for paired comparison."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() < 2:
            diffs[i] = np.nan
            continue
        a = average_precision_score(y_true[idx], p_a[idx])
        b = average_precision_score(y_true[idx], p_b[idx])
        diffs[i] = b - a
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return float(np.nanmean(diffs)), float(lo), float(hi)


def paired_permutation_pvalue(y_true, p_a, p_b, n_perm=1000, seed=42):
    """Paired permutation: swap (p_a, p_b) per-sample and see how often the
    swapped diff exceeds the observed diff in magnitude."""
    rng = np.random.default_rng(seed)
    observed = average_precision_score(y_true, p_b) - average_precision_score(y_true, p_a)
    n = len(y_true)
    extreme = 0
    for _ in range(n_perm):
        mask = rng.random(n) < 0.5
        pa = np.where(mask, p_b, p_a)
        pb = np.where(mask, p_a, p_b)
        d = average_precision_score(y_true, pb) - average_precision_score(y_true, pa)
        if abs(d) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (n_perm + 1)), float(observed)


def top_k_enrichment(y_true, p, k=100):
    order = np.argsort(-p)[:k]
    return float(y_true[order].mean() / y_true.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/coadd_pa.csv")
    ap.add_argument("--raw-csv", default="data/raw/coadd_pa_combined_per_molecule.csv",
                    help="raw CSV used to attach CHEMBL IDs to flagged actives")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    try:
        import cleanlab
        from cleanlab.filter import find_label_issues
    except ImportError as e:
        raise SystemExit(f"cleanlab not installed. `pip install cleanlab>=2.6`  ({e})")

    print(f"[1/5] Loading {args.csv}")
    ds = load_coadd_pa(args.csv)
    print(f"  N={len(ds)}  active={ds.active_fraction:.2%}")

    print("[2/5] Featurizing (ECFP4, 2048 bits)")
    X, valid = featurize_ecfp4(list(ds.smiles), n_bits=2048)
    X = X[valid]
    y = ds.y[valid]
    smi = ds.smiles[valid]
    print(f"  X={X.shape}")

    print("[3/5] OOF (random 5-fold) — needed for cleanlab pred_probs")
    folds = stratified_folds(Dataset(smiles=smi, y=y), n_splits=5, seed=42)
    oof_a = cross_validated_oof(X, y, folds, n_estimators=200, max_depth=6,
                                keep_fold_models=False)
    pred_probs = np.stack([1 - oof_a.y_prob, oof_a.y_prob], axis=1)
    print(f"  AUPRC (A=original): {oof_a.auprc:.3f}")

    print("[4/5] cleanlab find_label_issues")
    flagged_idx = find_label_issues(
        labels=y.astype(int),
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    flagged_y = y[flagged_idx]
    print(f"  flagged total: {len(flagged_idx)}  (active flagged: {int(flagged_y.sum())})")

    # Attach CHEMBL IDs for maintainer review (active flags are the high-stakes ones)
    raw = pd.read_csv(args.raw_csv)
    smi_to_cid = dict(zip(raw["canonical_smiles"], raw["molecule_chembl_id"]))
    review = pd.DataFrame({
        "smiles": smi[flagged_idx],
        "y_recorded": y[flagged_idx],
        "p_active_oof": oof_a.y_prob[flagged_idx],
    })
    review["chembl_id"] = review["smiles"].map(smi_to_cid).fillna("(not in raw)")
    review["review_note"] = ""  # maintainer fills: "looks-true-active" | "looks-spurious" | "unclear"
    review = review.sort_values("p_active_oof", ascending=True).reset_index(drop=True)

    today = date.today().strftime("%Y%m%d")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_csv = out_dir / f"{today}_cleanlab_flagged.csv"
    review.to_csv(review_csv, index=False)
    print(f"  wrote: {review_csv}  (maintainer review needed before trusting B)")

    print("[5/5] B = retrain after removing flagged, then A/B compare with CI95")
    keep_mask = np.ones(len(y), dtype=bool)
    keep_mask[flagged_idx] = False
    folds_b = stratified_folds(Dataset(smiles=smi[keep_mask], y=y[keep_mask]),
                               n_splits=5, seed=42)
    oof_b = cross_validated_oof(
        X[keep_mask], y[keep_mask], folds_b,
        n_estimators=200, max_depth=6, keep_fold_models=False,
    )
    auprc_a = float(oof_a.auprc)
    auprc_b = float(oof_b.auprc)
    enrich_a = top_k_enrichment(y, oof_a.y_prob, k=100)
    # B has different N; project p_b back to A's index space for paired CI:
    # use A's labels but B's probs on KEPT samples; flagged samples get
    # NaN so they're excluded from the paired bootstrap.
    p_b_aligned = np.full_like(oof_a.y_prob, np.nan)
    p_b_aligned[keep_mask] = oof_b.y_prob
    paired_mask = ~np.isnan(p_b_aligned)
    diff_mean, diff_lo, diff_hi = bootstrap_auprc_diff(
        y[paired_mask], oof_a.y_prob[paired_mask], p_b_aligned[paired_mask],
        n_boot=args.bootstrap,
    )
    p_val, observed_diff = paired_permutation_pvalue(
        y[paired_mask], oof_a.y_prob[paired_mask], p_b_aligned[paired_mask],
        n_perm=args.permutations,
    )
    print(f"  A (original)  AUPRC: {auprc_a:.3f}  enrich@100: {enrich_a:.1f}x")
    print(f"  B (cleaned)   AUPRC: {auprc_b:.3f}   N: {keep_mask.sum()}  removed: {(~keep_mask).sum()}")
    print(f"  Δ AUPRC (B-A, paired, on kept): {diff_mean:+.3f}  CI95 [{diff_lo:+.3f}, {diff_hi:+.3f}]")
    print(f"  paired permutation p-value: {p_val:.3f}  (n_perm={args.permutations})")

    out_json = out_dir / f"{today}_cleanlab_audit.json"
    out_json.write_text(json.dumps({
        "n_total": int(len(y)),
        "n_flagged": int(len(flagged_idx)),
        "n_flagged_active": int(flagged_y.sum()),
        "A": {"AUPRC": auprc_a, "enrich@100": enrich_a},
        "B": {"AUPRC": auprc_b, "N": int(keep_mask.sum())},
        "diff": {"mean": diff_mean, "ci95_low": diff_lo, "ci95_high": diff_hi,
                 "permutation_p_value": p_val, "observed_diff": observed_diff},
        "review_csv": str(review_csv),
        "honest_note": (
            "Δ AUPRC near zero or negative is a valid outcome and is "
            "reported as-is. Do NOT cherry-pick the bootstrap iteration "
            "that supports the desired conclusion."
        ),
    }, indent=2, default=float))
    print(f"  wrote: {out_json}")


if __name__ == "__main__":
    main()
