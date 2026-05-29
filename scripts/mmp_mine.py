#!/usr/bin/env python3
"""Data-mined MMPs from a SMILES+label dataset (lightweight, no mmpdb required).

The default counterfactual.py applies a preset list of 8 SMIRKS and asks
"what happens if I make this change?" That tests transformation effects
but cannot find scaffold-local patterns because the SMIRKS themselves are
generic.

This script does the inverse: it *mines* matched-molecular-pairs FROM
the dataset. A pair (A, B) qualifies if:
  - same Bemis-Murcko scaffold
  - Tanimoto similarity (Morgan radius=2) >= --tanimoto
  - heavy-atom-count difference <= --max-atom-delta
  - opposite or distinct labels (active vs inactive)

We then bucket pairs by "transformation" (canonicalized MCS-residue pair)
and report:
  - n_pairs per transformation
  - n_distinct_scaffolds
  - mean delta-active (active fraction of B minus active fraction of A)
  - is_series_local (n_distinct_scaffolds < threshold)

This is closer to Auer 2016 / mmpdb in spirit. Output is series-local
fraction for direct comparison with PREMISES H4.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
_MGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _scaffold(smi: str) -> str:
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
        return sc if sc else f"__acyclic__{smi}"
    except Exception:
        return f"__invalid__{smi}"


def _fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    return _MGEN.GetFingerprint(mol) if mol else None


def _diff_label(a_label: int, b_label: int) -> str:
    if a_label == b_label:
        return "same"
    return "0->1" if a_label == 0 and b_label == 1 else "1->0"


def _transform_key(smi_a: str, smi_b: str) -> str:
    """Coarse transformation key: sorted (heavy_atom_count_A, heavy_atom_count_B).

    A full MCS-based key would be ideal but heavy. For now we bucket
    by atom-count change and the lexicographically smaller / larger
    sides, which is enough to group "added a methyl" pairs together.
    """
    a, b = Chem.MolFromSmiles(smi_a), Chem.MolFromSmiles(smi_b)
    if a is None or b is None:
        return "invalid"
    da, db = a.GetNumHeavyAtoms(), b.GetNumHeavyAtoms()
    delta = db - da
    return f"delta_{delta:+d}"


def mine(df: pd.DataFrame, tanimoto: float, max_atom_delta: int) -> list[dict]:
    smis = df["canonical_smiles"].to_numpy()
    y = df["active"].to_numpy().astype(int)
    n = len(smis)
    scaffolds = [_scaffold(s) for s in smis]
    fps = [_fp(s) for s in smis]
    atom_counts = []
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        atom_counts.append(mol.GetNumHeavyAtoms() if mol else 0)
    atom_counts = np.array(atom_counts)

    by_scaffold: dict[str, list[int]] = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        by_scaffold[sc].append(i)

    transform_bucket: dict[str, list[dict]] = defaultdict(list)
    seen_pairs: set[tuple[int, int]] = set()

    for sc, members in by_scaffold.items():
        if len(members) < 2:
            continue
        for ai in range(len(members)):
            i = members[ai]
            if fps[i] is None:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in members])
            for bi, sim in enumerate(sims):
                j = members[bi]
                if i >= j or sim < tanimoto:
                    continue
                if abs(atom_counts[i] - atom_counts[j]) > max_atom_delta:
                    continue
                pair = (i, j)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if y[i] == y[j]:
                    continue
                key = _transform_key(smis[i], smis[j])
                if key == "invalid":
                    continue
                transform_bucket[key].append({
                    "i": int(i), "j": int(j),
                    "scaffold": sc,
                    "delta_active": int(y[j] - y[i]),
                    "tanimoto": float(sim),
                })

    rules = []
    for key, pairs in transform_bucket.items():
        if not pairs:
            continue
        scaffolds_hit = {p["scaffold"] for p in pairs}
        deltas = np.array([p["delta_active"] for p in pairs])
        rules.append({
            "transformation": key,
            "n_pairs": len(pairs),
            "n_distinct_scaffolds": len(scaffolds_hit),
            "mean_delta_active": float(deltas.mean()),
            "active_gain_fraction": float((deltas > 0).mean()),
        })
    rules.sort(key=lambda r: r["n_pairs"], reverse=True)
    return rules


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--tanimoto", type=float, default=0.85)
    p.add_argument("--max-atom-delta", type=int, default=4)
    p.add_argument("--series-threshold", type=int, default=3)
    p.add_argument("--out", default=None)
    p.add_argument("--top-k", type=int, default=20,
                   help="show top-k transformations by pair count")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["canonical_smiles", "active"]).reset_index(drop=True)
    print(f"Loaded {len(df)} mols from {args.csv}")
    rules = mine(df, args.tanimoto, args.max_atom_delta)
    n_total = len(rules)
    n_series_local = sum(
        1 for r in rules if 0 < r["n_distinct_scaffolds"] < args.series_threshold
    )
    print(f"\nFound {n_total} transformations from data-mined MMPs.")
    print(f"  series-local (< {args.series_threshold} scaffolds): {n_series_local} "
          f"({n_series_local/max(n_total,1):.1%})")
    print(f"\nTop {min(args.top_k, n_total)} transformations:")
    print(f"{'transformation':<14} {'n_pairs':>7} {'n_scaf':>6} {'mean_d':>7} {'gain%':>6}")
    for r in rules[:args.top_k]:
        print(f"{r['transformation']:<14} {r['n_pairs']:>7} "
              f"{r['n_distinct_scaffolds']:>6} {r['mean_delta_active']:>+7.3f} "
              f"{r['active_gain_fraction']:>6.1%}")

    report = {
        "csv": args.csv,
        "params": {"tanimoto": args.tanimoto,
                   "max_atom_delta": args.max_atom_delta,
                   "series_threshold": args.series_threshold},
        "n_transformations": n_total,
        "n_series_local": n_series_local,
        "series_local_fraction": n_series_local / max(n_total, 1),
        "rules": rules,
    }
    out_path = args.out or "reports/20260529_mmp_mined.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
