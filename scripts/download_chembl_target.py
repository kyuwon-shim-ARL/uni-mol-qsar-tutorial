#!/usr/bin/env python3
"""Download a single-target activity dataset from ChEMBL by target_chembl_id.

Pulls all bioactivities for a given ChEMBL target (e.g. CHEMBL5145 = BRAF),
keeps the rows that have a numeric `pchembl_value`, and applies a binary
active/inactive label using a pChEMBL threshold (default 6.0, i.e.
IC50/Ki/Kd <= 1 uM).

Output schema matches the existing tutorial's processed CSV:
    canonical_smiles, active

So the file can be fed directly to scripts/prepare_data.py or to
src/qsar_tutorial/data.load_coadd_pa() (which just needs those two cols).

Usage:
    # BRAF, pChEMBL >= 6
    python scripts/download_chembl_target.py --target CHEMBL5145 --name braf

    # EGFR, stricter cutoff
    python scripts/download_chembl_target.py --target CHEMBL203 --name egfr \\
        --pchembl-cutoff 7.0

Writes to data/raw/:
    {name}_activities.csv          all retrieved rows (audit trail)
    {name}_per_molecule.csv        canonical_smiles, active (ML-ready)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

KEEP_COLS = [
    "molecule_chembl_id", "canonical_smiles",
    "standard_type", "standard_value", "standard_units", "standard_relation",
    "pchembl_value",
    "assay_chembl_id", "assay_description", "assay_type",
    "target_chembl_id", "target_pref_name", "target_organism",
    "activity_id", "document_chembl_id", "document_year",
    "data_validity_comment", "potential_duplicate",
]


def _http_json(url: str, *, timeout: int = 45, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt + 1 == retries:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}] {e}; wait {wait}s", file=sys.stderr)
            time.sleep(wait)
    return {}


def _count_records(target_id: str, *, require_pchembl: bool = True) -> int:
    params = {
        "target_chembl_id": target_id,
        "format": "json", "limit": 1, "offset": 0,
    }
    if require_pchembl:
        params["pchembl_value__isnull"] = "false"
    url = f"{CHEMBL_BASE}/activity?" + urllib.parse.urlencode(params)
    return int(_http_json(url, timeout=30)["page_meta"]["total_count"])


def _fetch_page(target_id: str, offset: int, limit: int = 1000,
                *, require_pchembl: bool = True) -> list[dict]:
    params = {
        "target_chembl_id": target_id,
        "format": "json", "limit": limit, "offset": offset,
    }
    if require_pchembl:
        params["pchembl_value__isnull"] = "false"
    url = f"{CHEMBL_BASE}/activity?" + urllib.parse.urlencode(params)
    return _http_json(url).get("activities", [])


def download_target(target_id: str, *, require_pchembl: bool = True) -> pd.DataFrame:
    total = _count_records(target_id, require_pchembl=require_pchembl)
    label = "with pchembl_value" if require_pchembl else "(all, inc. null pchembl)"
    print(f"  {target_id}: {total} activity records {label}")
    if total == 0:
        return pd.DataFrame(columns=KEEP_COLS)
    offsets = list(range(0, total, 1000))
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(_fetch_page, target_id, off,
                             require_pchembl=require_pchembl) for off in offsets]
        for fut in as_completed(futures):
            records.extend(fut.result())
    df = pd.DataFrame(records)
    df = df[[c for c in KEEP_COLS if c in df.columns]]
    df = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    return df


def per_molecule_with_null_inactives(
    df: pd.DataFrame, pchembl_cutoff: float
) -> pd.DataFrame:
    """Variant that retains null-pchembl rows: a molecule whose max pchembl
    is below cutoff OR all-null counts as inactive. This boosts the pool of
    inactives and partly mitigates ChEMBL publication bias (where weak
    binders rarely get a pchembl_value computed).

    Caveat: rows with null pchembl may be qualitative inhibition assays at
    a fixed concentration; treating them as inactive assumes "if it were
    potent, someone would have computed pchembl." This is a working
    approximation, not ground truth.
    """
    pchembl = pd.to_numeric(df["pchembl_value"], errors="coerce")
    year = pd.to_numeric(df.get("document_year"), errors="coerce")
    df = df.assign(_pchembl=pchembl, _year=year)
    # Filter: row must have either a numeric pchembl OR a non-null standard_value
    # (i.e. there must be some measurement; pure metadata rows excluded).
    has_pchembl = df["_pchembl"].notna()
    has_value = pd.to_numeric(df.get("standard_value"), errors="coerce").notna()
    df = df[has_pchembl | has_value]
    g = (
        df.groupby("molecule_chembl_id", as_index=False)
          .agg(
              canonical_smiles=("canonical_smiles", "first"),
              max_pchembl=("_pchembl", "max"),
              n_pchembl_measurements=("_pchembl", "count"),
              n_total_measurements=("_pchembl", "size"),
              min_document_year=("_year", "min"),
          )
    )
    # Active iff max_pchembl >= cutoff. NaN max_pchembl -> not active (inactive).
    g["active"] = (g["max_pchembl"].fillna(-np.inf) >= pchembl_cutoff).astype(int)
    return g[["molecule_chembl_id", "canonical_smiles", "max_pchembl",
              "n_pchembl_measurements", "n_total_measurements",
              "min_document_year", "active"]]


def per_molecule(df: pd.DataFrame, pchembl_cutoff: float) -> pd.DataFrame:
    """Aggregate to one row per molecule. active = max(pchembl) >= cutoff.

    Censored "<" rows can only argue against activity; "=" / ">" rows can
    establish activity. We keep all rows for the max() since pchembl_value
    is already a numeric upper estimate.

    Also retains `min_document_year` (earliest publication of any
    measurement for this molecule). Used by data.time_split_indices for
    chronological train/test splits.
    """
    pchembl = pd.to_numeric(df["pchembl_value"], errors="coerce")
    year = pd.to_numeric(df.get("document_year"), errors="coerce")
    df = df.assign(_pchembl=pchembl, _year=year).dropna(subset=["_pchembl"])
    g = (
        df.groupby("molecule_chembl_id", as_index=False)
          .agg(
              canonical_smiles=("canonical_smiles", "first"),
              max_pchembl=("_pchembl", "max"),
              n_measurements=("_pchembl", "size"),
              min_document_year=("_year", "min"),
          )
    )
    g["active"] = (g["max_pchembl"] >= pchembl_cutoff).astype(int)
    return g[["molecule_chembl_id", "canonical_smiles", "max_pchembl",
              "n_measurements", "min_document_year", "active"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", required=True,
                        help="ChEMBL target id, e.g. CHEMBL5145 (BRAF).")
    parser.add_argument("--name", required=True,
                        help="Short slug used in output filenames, e.g. braf.")
    parser.add_argument("--pchembl-cutoff", type=float, default=6.0,
                        help="pChEMBL threshold for active=1 (default 6.0 = 1 uM).")
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if outputs already exist.")
    parser.add_argument(
        "--include-null-pchembl-as-inactive", action="store_true",
        help=(
            "Publication-bias mitigation: also fetch rows with null "
            "pchembl_value but a non-null standard_value, and treat them as "
            "inactive. Boosts the inactive pool. See per_molecule_with_null_inactives() docstring."
        ),
    )
    args = parser.parse_args(argv)

    raw_dir = args.out_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = args.name.lower()
    activities_csv = raw_dir / f"{name}_activities.csv"
    per_mol_csv = raw_dir / f"{name}_per_molecule.csv"

    if not args.force and activities_csv.exists() and per_mol_csv.exists():
        print(f"Outputs already present for '{name}'. Use --force to refetch.")
        return 0

    print(f"[1/2] Downloading {args.target} bioactivities from ChEMBL")
    df = download_target(
        args.target,
        require_pchembl=not args.include_null_pchembl_as_inactive,
    )
    if df.empty:
        print(f"No data for {args.target}.", file=sys.stderr)
        return 1
    df.to_csv(activities_csv, index=False)
    print(f"  -> {activities_csv.name} ({len(df)} rows)")

    print(f"[2/2] Aggregating per molecule (pchembl >= {args.pchembl_cutoff})")
    if args.include_null_pchembl_as_inactive:
        g = per_molecule_with_null_inactives(df, args.pchembl_cutoff)
    else:
        g = per_molecule(df, args.pchembl_cutoff)
    g.to_csv(per_mol_csv, index=False)
    n_active = int(g["active"].sum())
    print(f"  -> {per_mol_csv.name} ({len(g)} molecules, "
          f"{n_active} active = {g['active'].mean():.2%})")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
