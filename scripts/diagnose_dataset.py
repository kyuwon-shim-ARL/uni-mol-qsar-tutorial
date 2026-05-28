#!/usr/bin/env python3
"""Diagnose whether a SMILES+label dataset is amenable to the interpretable
QSAR pipeline (Uni-Mol + XGBoost + SHAP + SAE + MMP).

Report card includes:
  1. Size / active fraction / SMILES validity
  2. Bemis-Murcko scaffold count + top-N series dominance
  3. pChEMBL distribution + suggested cutoffs (if max_pchembl column present)
  4. Activity-cliff prevalence: % of molecules with a near-neighbor
     (Morgan/Tanimoto >= 0.85) carrying the opposite label
  5. document_year distribution (if column present) -- time-split feasibility

Use this BEFORE running the full pipeline to know what to expect:
  - Many scaffolds + cliff% low      -> "diverse phenotypic-like"   -> CO-ADD-style
  - Few scaffolds  + cliff% high     -> "SAR-dense single-target"   -> BRAF-style
  - The two regimes give very different interpretability outputs.

Usage:
    python scripts/diagnose_dataset.py data/processed/braf.csv
    python scripts/diagnose_dataset.py data/processed/coadd_pa.csv --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
_MGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _scaffold(smi: str) -> str | None:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return None


def _fp(smi: str) -> object | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return _MGEN.GetFingerprint(mol)


def cliff_fraction(
    smiles: np.ndarray, y: np.ndarray, *, threshold: float = 0.85
) -> tuple[float, int]:
    """Fraction of molecules with at least one near-neighbor of opposite label.

    Definition (coarse MMP proxy): for each molecule M with fingerprint f_M,
    if there exists another molecule M' with Tanimoto(f_M, f_M') >= threshold
    AND label(M) != label(M'), then M is "on a cliff".

    O(N^2) but vectorized via BulkTanimoto. Manageable up to ~10K molecules.
    Returns (fraction, count_on_cliff).
    """
    fps = [_fp(s) for s in smiles]
    valid = [i for i, f in enumerate(fps) if f is not None]
    fps = [fps[i] for i in valid]
    y = y[valid]
    n = len(fps)
    if n < 2:
        return 0.0, 0
    on_cliff = np.zeros(n, dtype=bool)
    for i in range(n):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        sims[i] = 0.0
        near = sims >= threshold
        if not near.any():
            continue
        opp = y[near] != y[i]
        if opp.any():
            on_cliff[i] = True
    return float(on_cliff.mean()), int(on_cliff.sum())


def scaffold_stats(smiles: np.ndarray, y: np.ndarray, top_n: int = 10) -> dict:
    scaffolds = [_scaffold(s) for s in smiles]
    scaffolds = [s if s else f"__acyclic__{i}" for i, s in enumerate(scaffolds)]
    counter = Counter(scaffolds)
    n_unique = len(counter)
    top = counter.most_common(top_n)
    top_share = sum(c for _, c in top) / len(scaffolds)
    largest_share = top[0][1] / len(scaffolds) if top else 0.0
    # Active-fraction within the top scaffold (series-signal sanity check)
    top_scaf = top[0][0] if top else None
    if top_scaf:
        mask = np.array([s == top_scaf for s in scaffolds])
        top_active = float(y[mask].mean()) if mask.any() else 0.0
    else:
        top_active = 0.0
    return {
        "n_unique_scaffolds": n_unique,
        "n_molecules": len(scaffolds),
        "molecules_per_scaffold_median": float(
            np.median(list(counter.values()))
        ),
        "molecules_per_scaffold_max": int(max(counter.values())),
        "top1_share": float(largest_share),
        "top10_share": float(top_share),
        "top1_active_fraction": top_active,
    }


def diagnose(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if not {"canonical_smiles", "active"}.issubset(df.columns):
        raise ValueError("CSV must have canonical_smiles and active columns.")
    df = df.dropna(subset=["canonical_smiles", "active"]).reset_index(drop=True)
    df["active"] = df["active"].astype(int)

    smi = df["canonical_smiles"].to_numpy()
    y = df["active"].to_numpy().astype(int)

    valid = np.array([Chem.MolFromSmiles(s) is not None for s in smi])
    n_invalid = int((~valid).sum())
    smi = smi[valid]
    y = y[valid]

    out: dict = {
        "csv": str(csv_path),
        "n_total_rows": int(len(df)),
        "n_invalid_smiles": n_invalid,
        "n_valid": int(len(smi)),
        "active_fraction": float(y.mean()),
        "n_active": int(y.sum()),
        "n_inactive": int(len(y) - y.sum()),
    }

    out["scaffolds"] = scaffold_stats(smi, y)

    if "max_pchembl" in df.columns or "pchembl_value" in df.columns:
        col = "max_pchembl" if "max_pchembl" in df.columns else "pchembl_value"
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        out["pchembl"] = {
            "column": col,
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "median": float(vals.median()),
            "max": float(vals.max()),
            "active_pct_at_cutoff": {
                str(c): float((vals >= c).mean())
                for c in [5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
            },
        }

    if "document_year" in df.columns or "min_document_year" in df.columns:
        col = (
            "min_document_year"
            if "min_document_year" in df.columns
            else "document_year"
        )
        years = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        if len(years) > 0:
            out["years"] = {
                "column": col,
                "n_with_year": int(len(years)),
                "min": int(years.min()),
                "max": int(years.max()),
                "median": int(years.median()),
                "p25": int(years.quantile(0.25)),
                "p75": int(years.quantile(0.75)),
            }

    cap = 6000
    if len(smi) > cap:
        idx = np.random.default_rng(42).choice(len(smi), size=cap, replace=False)
        smi_s, y_s = smi[idx], y[idx]
        cliff_subsample = True
    else:
        smi_s, y_s = smi, y
        cliff_subsample = False
    cliff_frac, cliff_n = cliff_fraction(smi_s, y_s, threshold=0.85)
    out["activity_cliffs"] = {
        "tanimoto_threshold": 0.85,
        "fraction_on_cliff": cliff_frac,
        "n_on_cliff": cliff_n,
        "subsampled": cliff_subsample,
        "subsample_size": int(len(smi_s)) if cliff_subsample else None,
    }

    return out


def _verdict(r: dict) -> list[str]:
    """One-line per concern. No sugar-coating."""
    lines = []
    sc = r["scaffolds"]
    cliff = r["activity_cliffs"]["fraction_on_cliff"]
    af = r["active_fraction"]

    if sc["top1_share"] >= 0.10:
        lines.append(
            f"[!] series-dominance: top-1 scaffold covers "
            f"{sc['top1_share']:.1%} of molecules "
            f"(active% within = {sc['top1_active_fraction']:.1%}) -- "
            f"MMP rules from this scaffold may be series-specific, not general."
        )
    if sc["top10_share"] >= 0.30:
        lines.append(
            f"[!] top-10 scaffolds cover {sc['top10_share']:.1%} -- "
            f"chemical diversity narrow; SAE descriptor recovery R^2 expected lower."
        )
    if cliff >= 0.30:
        lines.append(
            f"[!] activity cliffs: {cliff:.1%} of molecules have a near-neighbor "
            f"(Tanimoto>=0.85) with opposite label -- random K-fold WILL overstate "
            f"AUC; use scaffold split."
        )
    if af < 0.10 or af > 0.90:
        lines.append(
            f"[!] severe class imbalance: active fraction = {af:.1%}. "
            f"Reconsider cutoff or report AUPRC (not just AUC)."
        )
    if not lines:
        lines.append(
            "[ok] no major red flags. Pipeline assumptions look defensible."
        )
    return lines


def print_human(r: dict) -> None:
    print(f"\n=== Dataset diagnostic: {r['csv']} ===\n")
    print(f"Rows total      : {r['n_total_rows']}")
    print(f"Valid SMILES    : {r['n_valid']}  (invalid: {r['n_invalid_smiles']})")
    print(
        f"Labels          : {r['n_active']} active / "
        f"{r['n_inactive']} inactive  ({r['active_fraction']:.1%} active)"
    )
    print()
    sc = r["scaffolds"]
    print("Bemis-Murcko scaffolds:")
    print(f"  unique          : {sc['n_unique_scaffolds']}")
    print(f"  median per scaf : {sc['molecules_per_scaffold_median']:.1f}")
    print(f"  max in one scaf : {sc['molecules_per_scaffold_max']}")
    print(
        f"  top1 share      : {sc['top1_share']:.1%} "
        f"(active% within = {sc['top1_active_fraction']:.1%})"
    )
    print(f"  top10 share     : {sc['top10_share']:.1%}")
    print()
    if "pchembl" in r:
        p = r["pchembl"]
        print(f"pChEMBL ({p['column']}):")
        print(
            f"  N={p['n']}  mean={p['mean']:.2f}  median={p['median']:.2f}  "
            f"std={p['std']:.2f}  range=[{p['min']:.1f}, {p['max']:.1f}]"
        )
        print("  Active% at cutoff:")
        for c, pct in p["active_pct_at_cutoff"].items():
            print(f"    >= {c}: {pct:.1%}")
        print()
    if "years" in r:
        y = r["years"]
        print(f"Document years ({y['column']}):")
        print(
            f"  N={y['n_with_year']}  min={y['min']}  p25={y['p25']}  "
            f"median={y['median']}  p75={y['p75']}  max={y['max']}"
        )
        print(f"  -> time-split feasible (e.g. train <= {y['median']}, test > {y['median']})")
        print()
    ac = r["activity_cliffs"]
    note = f" (subsampled {ac['subsample_size']})" if ac["subsampled"] else ""
    print(
        f"Activity cliffs (Tanimoto>={ac['tanimoto_threshold']}, opp. label){note}:"
    )
    print(
        f"  {ac['fraction_on_cliff']:.1%} of molecules on a cliff "
        f"(N={ac['n_on_cliff']})"
    )
    print()
    print("Verdict:")
    for line in _verdict(r):
        print(f"  {line}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="CSV with canonical_smiles + active.")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON only (no human-readable output).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write JSON report.")
    args = parser.parse_args(argv)

    report = diagnose(args.csv)
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        print_human(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        if not args.json:
            print(f"JSON report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
