"""T1 + T4 deliverable — random vs scaffold split with per-fold SHAP stability.

Slim runner (faster than run_full_pipeline.py — no SAE, no MMP). Produces:
  - reports/{YYYYMMDD}_split-compare_{featurizer}.html
  - reports/{YYYYMMDD}_split-compare_{featurizer}.json   (fold counts + AUCs)

ECFP4 path runs CPU-only and is the GPU-free deliverable. Uni-Mol path
requires the runpod-mcp preflight gate to PASS before launch.

Usage:
    python examples/run_split_comparison.py --features ecfp4
    python examples/run_split_comparison.py --features unimol     # GPU
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from qsar_tutorial.data import (
    Dataset,
    fold_label_counts,
    load_coadd_pa,
    scaffold_folds,
    stratified_folds,
)
from qsar_tutorial.featurizer import featurize_ecfp4
from qsar_tutorial.model import cross_validated_oof
from qsar_tutorial.report import ReportPayload, render_report
from qsar_tutorial.shap_layer import per_fold_top_features, stability_summary


def featurize(smiles, kind: str):
    if kind == "ecfp4":
        X, valid = featurize_ecfp4(list(smiles), n_bits=2048)
        names = [f"ecfp_{i}" for i in range(X.shape[1])]
        return X, valid, names
    if kind == "unimol":
        from qsar_tutorial.featurizer import UniMolFeaturizer

        feat = UniMolFeaturizer()
        X, valid = feat.featurize(list(smiles))
        names = [f"unimol_{i}" for i in range(X.shape[1])]
        return X, valid, names
    raise ValueError(f"unknown featurizer: {kind}")


def run_one_split(name: str, X, y, smiles, splitter, n_estimators, max_depth, top_k):
    """Returns (split_summary_row, stability_rows, fold_counts)."""
    print(f"  [{name}] generating folds...")
    folds = splitter(Dataset(smiles=smiles, y=y), n_splits=5, seed=42)
    counts = fold_label_counts(Dataset(smiles=smiles, y=y), folds)
    print(f"  [{name}] fold counts: {[c['n_active_val'] for c in counts]} actives in val")

    oof = cross_validated_oof(
        X, y, folds,
        n_estimators=n_estimators, max_depth=max_depth,
        keep_fold_models=True,
    )
    fold_aucs = np.array([f["AUC"] for f in oof.per_fold])
    summary = oof.summary()
    row = {
        "split": name,
        "AUC": summary["AUC"],
        "AUPRC": summary["AUPRC"],
        "MCC@0.5": summary["MCC@0.5"],
        "fold_auc_std": float(np.nanstd(fold_aucs)),
        "fold_aucs": fold_aucs.tolist(),
        "fold_counts": counts,
    }
    print(f"  [{name}] AUC={row['AUC']:.3f}  AUPRC={row['AUPRC']:.3f}  fold_std={row['fold_auc_std']:.3f}")

    print(f"  [{name}] per-fold SHAP top-{top_k} ...")
    per_fold = per_fold_top_features(
        oof.fold_models, X, oof.fold_indices,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
        k=top_k, explain_on="train",
    )
    stab = stability_summary(per_fold)
    print(f"  [{name}] stability rows: {len(stab)} unique features across folds")
    return row, stab, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/coadd_pa.csv")
    p.add_argument("--features", choices=["ecfp4", "unimol"], default="ecfp4")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out-dir", default="reports")
    args = p.parse_args()

    print(f"[1/4] Loading {args.csv}")
    ds = load_coadd_pa(args.csv)
    print(f"  N={len(ds)}  active={ds.active_fraction:.2%}")

    print(f"[2/4] Featurizing ({args.features})")
    X, valid, names = featurize(ds.smiles, args.features)
    X = X[valid]
    y = ds.y[valid]
    smi_valid = ds.smiles[valid]
    print(f"  X={X.shape}  y_active={y.mean():.2%}")

    print("[3/4] Running random and scaffold splits in sequence")
    rows = []
    stability_all = {}
    counts_all = {}
    for split_name, splitter in [("random", stratified_folds), ("scaffold", scaffold_folds)]:
        row, stab, counts = run_one_split(
            split_name, X, y, smi_valid, splitter,
            args.n_estimators, args.max_depth, args.top_k,
        )
        rows.append(row)
        stability_all[split_name] = stab
        counts_all[split_name] = counts

    gap_auc = rows[0]["AUC"] - rows[1]["AUC"]
    gap_auprc = rows[0]["AUPRC"] - rows[1]["AUPRC"]
    print(f"\n  gap (random - scaffold):  AUC={gap_auc:+.3f}  AUPRC={gap_auprc:+.3f}")

    print(f"[4/4] Rendering report")
    today = date.today().strftime("%Y%m%d")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / f"{today}_split-compare_{args.features}.html"
    out_json = out_dir / f"{today}_split-compare_{args.features}.json"

    # Stability shown for SCAFFOLD split (more honest baseline). Both stored
    # in JSON so users can inspect random-split stability too.
    payload = ReportPayload(
        title=f"CO-ADD PA — random vs scaffold split ({args.features})",
        dataset_name="CO-ADD P. aeruginosa (public ChEMBL mirror)",
        n_total=len(y),
        metrics={"AUC": rows[1]["AUC"], "AUPRC": rows[1]["AUPRC"], "MCC@0.5": rows[1]["MCC@0.5"]},
        top_shap=[(r["feature"], r["mean_importance"]) for r in stability_all["scaffold"][:args.top_k]],
        sae={"latent_dim": 0, "dead_ratio": 0.0, "r2_median": 0.0, "r2_above_05": 0, "r2_total": 0},
        mmp_rows=[],
        stability_rows=stability_all["scaffold"][:args.top_k],
        stability_n_folds=5,
        split_comparison=rows,
    )
    render_report(payload, out_html)

    out_json.write_text(json.dumps({
        "featurizer": args.features,
        "n_total": int(len(y)),
        "split_comparison": rows,
        "stability": stability_all,
        "fold_counts": counts_all,
        "gap": {"AUC": gap_auc, "AUPRC": gap_auprc},
    }, indent=2, default=float))
    print(f"  wrote: {out_html}")
    print(f"  wrote: {out_json}")


if __name__ == "__main__":
    main()
