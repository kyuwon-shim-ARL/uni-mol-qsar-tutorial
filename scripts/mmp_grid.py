#!/usr/bin/env python3
"""MMP rule ablation grid: inactive subset size × series-local threshold.

Tests whether the series-local fraction (H4 in PREMISES.md) depends on
inactive-subset sampling depth and the chosen scaffold threshold.

Output: reports/{date}_mmp_grid_{target}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsar_tutorial.counterfactual import scan
from qsar_tutorial.data import load_coadd_pa
from qsar_tutorial.featurizer import featurize_ecfp4
from qsar_tutorial.model import build_classifier, fit_with_class_weight


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=True, help="slug for output filename")
    p.add_argument("--subsets", type=int, nargs="+", default=[500, 1500, 0],
                   help="inactive subset sizes; 0 = all inactives")
    p.add_argument("--thresholds", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ds = load_coadd_pa(args.csv)
    X, valid = featurize_ecfp4(list(ds.smiles), n_bits=2048)
    X, y, smi = X[valid], ds.y[valid], ds.smiles[valid]

    clf = build_classifier(n_estimators=200, max_depth=6)
    fit_with_class_weight(clf, X, y)

    inactive = smi[y == 0].tolist()
    n_inactive = len(inactive)

    def score_fn(smis):
        Xs, vs = featurize_ecfp4(smis, n_bits=X.shape[1])
        proba = clf.predict_proba(Xs)[:, 1]
        proba[~vs] = 0.0
        return proba

    cells = []
    for subset in args.subsets:
        n_par = n_inactive if subset == 0 else min(subset, n_inactive)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_inactive, size=n_par, replace=False)
        parents = [inactive[i] for i in idx]
        print(f"[subset={n_par}] scanning {len(parents)} parents...")
        rules = scan(parents, score_fn)
        for thr in args.thresholds:
            series_local = sum(
                1 for r in rules if 0 < r.n_distinct_scaffolds < thr
            )
            cross_series = sum(
                1 for r in rules if r.n_distinct_scaffolds >= thr
            )
            cells.append({
                "inactive_subset": n_par,
                "series_local_threshold": thr,
                "n_rules_total": len(rules),
                "n_series_local": series_local,
                "n_cross_series": cross_series,
                "series_local_fraction": series_local / max(len(rules), 1),
                "rule_scaffold_counts": [
                    {"name": r.name, "n_scaffolds": r.n_distinct_scaffolds,
                     "delta_p_mean": r.delta_p_mean,
                     "ci_excl_zero": (r.delta_p_ci95[0] > 0 or r.delta_p_ci95[1] < 0)}
                    for r in rules
                ],
            })

    report = {
        "target": args.target,
        "n_total_inactive": n_inactive,
        "subsets_tested": args.subsets,
        "thresholds_tested": args.thresholds,
        "cells": cells,
    }
    out_path = args.out or f"reports/20260529_{args.target}_mmp_grid.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {out_path}")
    print("\nSummary table:")
    print(f"{'subset':>8} {'thr':>4} {'rules':>6} {'series':>8} {'cross':>6} {'frac':>6}")
    for c in cells:
        print(f"{c['inactive_subset']:>8} {c['series_local_threshold']:>4} "
              f"{c['n_rules_total']:>6} {c['n_series_local']:>8} "
              f"{c['n_cross_series']:>6} {c['series_local_fraction']:>6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
