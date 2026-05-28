#!/usr/bin/env python3
"""Download a target's binding-affinity ligands from BindingDB.

Used to obtain an *external* validation set for a ChEMBL-trained model.
BindingDB curates from primary literature independently of ChEMBL, so
the overlap is partial — the non-overlap is a genuine held-out set.

BindingDB REST endpoint:
    https://www.bindingdb.org/rest/getLigandsByUniprots
        ?uniprot=<UniProt>&cutoff=<nM_threshold>&code=0

Returns JSON. Fields per record: monomerid, smile (NOT canonical),
affinity_type (Ki|Kd|IC50|EC50), affinity (nM string).

Output schema matches scripts/download_chembl_target.py:
    canonical_smiles, active   (+ provenance columns)

Usage:
    # BRAF, default cutoff 10 nM for active
    python scripts/download_bindingdb_target.py \\
        --uniprot P15056 --name braf_bindingdb \\
        --subtract data/raw/braf_per_molecule.csv

The --subtract flag removes any canonical SMILES already present in the
training set so the file is a true external test (no overlap).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

BINDINGDB_REST = "https://www.bindingdb.org/rest/getLigandsByUniprots"


def fetch(uniprot: str, cutoff_nm: int) -> list[dict]:
    """Fetch affinity records. cutoff_nm is the *upper bound* — BindingDB
    returns ligands with affinity <= cutoff_nm.

    To approximate "all data", pass a large cutoff (e.g. 1e8 nM). Note
    that BindingDB's response can be large; expect a single-shot pull
    in the 10s of MB range for well-studied targets.
    """
    params = {"uniprot": uniprot, "cutoff": str(cutoff_nm), "code": "0"}
    url = BINDINGDB_REST + "?" + urllib.parse.urlencode(params)
    print(f"  fetching {url}")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read())
    # The top-level key has a typo ("Linds") in the live API — handle both.
    payload = raw.get("getLindsByUniprotsResponse") or raw.get(
        "getLigandsByUniprotsResponse"
    )
    if not payload:
        raise RuntimeError(f"Unexpected BindingDB response shape: {list(raw)}")
    return payload.get("affinities", [])


def to_dataframe(records: list[dict], active_cutoff_nm: float) -> pd.DataFrame:
    rows = []
    for r in records:
        smi = r.get("smile") or ""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        try:
            aff = float(r.get("affinity"))
        except (TypeError, ValueError):
            continue
        rows.append({
            "canonical_smiles": canon,
            "monomerid": r.get("monomerid"),
            "affinity_type": r.get("affinity_type"),
            "affinity_nm": aff,
            "pmid": r.get("pmid"),
            "doi": r.get("doi"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = (
        df.groupby("canonical_smiles", as_index=False)
          .agg(
              monomerid=("monomerid", "first"),
              affinity_type=("affinity_type", "first"),
              min_affinity_nm=("affinity_nm", "min"),
              n_measurements=("affinity_nm", "size"),
              pmid=("pmid", "first"),
              doi=("doi", "first"),
          )
    )
    g["active"] = (g["min_affinity_nm"] <= active_cutoff_nm).astype(int)
    return g


def subtract_overlap(ext: pd.DataFrame, ref_csv: Path) -> pd.DataFrame:
    ref = pd.read_csv(ref_csv)
    if "canonical_smiles" not in ref.columns:
        print(f"  [warn] {ref_csv} has no canonical_smiles; skipping subtract.")
        return ext

    canon = []
    for smi in ref["canonical_smiles"].fillna(""):
        mol = Chem.MolFromSmiles(smi)
        canon.append(Chem.MolToSmiles(mol, canonical=True) if mol else None)
    ref_set = set(c for c in canon if c)
    before = len(ext)
    ext = ext[~ext["canonical_smiles"].isin(ref_set)].reset_index(drop=True)
    print(
        f"  subtracted {before - len(ext)} overlap mols "
        f"({(before - len(ext)) / max(before, 1):.1%}) "
        f"-> {len(ext)} external-only mols"
    )
    return ext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--uniprot", required=True,
                        help="UniProt accession (e.g. P15056 for BRAF).")
    parser.add_argument("--name", required=True,
                        help="Slug for output filenames, e.g. braf_bindingdb.")
    parser.add_argument(
        "--active-cutoff-nm", type=float, default=10.0,
        help="affinity <= this (nM) => active. Default 10 nM (= pChEMBL 8).",
    )
    parser.add_argument(
        "--query-cutoff-nm", type=int, default=10_000_000,
        help="BindingDB-side affinity upper bound for query (default 1e7 nM).",
    )
    parser.add_argument(
        "--subtract", type=Path, default=None,
        help=(
            "Optional CSV of canonical_smiles to subtract from the result "
            "(removes overlap with training set)."
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "external",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] Fetching BindingDB ligands for {args.uniprot}")
    records = fetch(args.uniprot, args.query_cutoff_nm)
    print(f"  {len(records)} raw affinity records")

    print(f"[2/3] Canonicalizing + aggregating (active <= {args.active_cutoff_nm} nM)")
    df = to_dataframe(records, args.active_cutoff_nm)
    if df.empty:
        print("  no usable records; abort.", file=sys.stderr)
        return 1
    print(f"  {len(df)} unique molecules, {int(df.active.sum())} active "
          f"({df.active.mean():.1%})")

    if args.subtract:
        print(f"[2.5/3] Subtracting overlap with {args.subtract}")
        df = subtract_overlap(df, args.subtract)
        if df.empty:
            print("  no external-only molecules remain.", file=sys.stderr)
            return 1
        print(f"  after subtract: {len(df)} molecules, "
              f"{int(df.active.sum())} active ({df.active.mean():.1%})")

    out = args.out_dir / f"{args.name}.csv"
    df.to_csv(out, index=False)
    print(f"[3/3] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
