#!/usr/bin/env python3
"""Time-split AUC sweep across multiple year cutoffs.

Lighter than run_full_pipeline.py — only trains XGBoost and measures
time-split AUC (no SAE/SHAP/MMP). Used to produce the "AUC vs cutoff"
curve referenced in EXPECTED_OUTPUTS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsar_tutorial.data import (
    Dataset, load_coadd_pa, load_years, time_split_indices,
)
from qsar_tutorial.featurizer import featurize_ecfp4
from qsar_tutorial.model import build_classifier, fit_with_class_weight


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--year-csv", required=True)
    p.add_argument("--cutoffs", type=int, nargs="+",
                   default=[2012, 2014, 2016, 2018])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ds = load_coadd_pa(args.csv)
    years = load_years(args.year_csv, ds.smiles)
    X, valid = featurize_ecfp4(list(ds.smiles), n_bits=2048)
    X, y, years_v = X[valid], ds.y[valid], years[valid]

    rows = []
    for cutoff in args.cutoffs:
        tr, te = time_split_indices(years_v, cutoff)
        if len(tr) < 20 or len(te) < 20 or len(np.unique(y[te])) < 2:
            rows.append({
                "cutoff": cutoff,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "AUC": None,
                "AUPRC": None,
                "skip_reason": "degenerate split",
            })
            continue
        clf = build_classifier(n_estimators=200, max_depth=6)
        fit_with_class_weight(clf, X[tr], y[tr])
        p_te = clf.predict_proba(X[te])[:, 1]
        rows.append({
            "cutoff": cutoff,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_active_pct": float(y[tr].mean()),
            "test_active_pct": float(y[te].mean()),
            "AUC": float(roc_auc_score(y[te], p_te)),
            "AUPRC": float(average_precision_score(y[te], p_te)),
        })

    out_path = args.out or "reports/20260529_time_sweep.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"csv": args.csv, "rows": rows}, f, indent=2)

    print(f"\n{'cutoff':>7} {'n_train':>8} {'n_test':>7} {'AUC':>6} {'AUPRC':>6}")
    for r in rows:
        auc = f"{r['AUC']:.3f}" if r["AUC"] is not None else "skip"
        auprc = f"{r['AUPRC']:.3f}" if r.get("AUPRC") is not None else "—"
        print(f"{r['cutoff']:>7} {r['n_train']:>8} {r['n_test']:>7} {auc:>6} {auprc:>6}")
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
