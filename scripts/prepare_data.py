"""Prepare the CO-ADD P. aeruginosa public dataset for the tutorial.

The "CO-ADD" PA dataset this tutorial uses is hosted in ChEMBL
(src_id=40, src_short_name=COADD) and is accessible via the public ChEMBL
REST API. No registration is required.

If  data/raw/coadd_pa_combined_per_molecule.csv  is missing this script
delegates to  scripts/download_chembl_coadd.py  to fetch the source assays
(~5 minutes, ~25 MB) and produce the four lineage CSVs in  data/raw/.

Then this script canonicalizes SMILES with RDKit, de-duplicates, and writes:
    data/processed/coadd_pa.csv  with columns  canonical_smiles, active
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def canonicalize(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW / "coadd_pa_combined_per_molecule.csv",
        help="Path to a CO-ADD PA CSV with 'canonical_smiles' and 'active' columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED / "coadd_pa.csv",
    )
    parser.add_argument(
        "--smiles-col", default="canonical_smiles",
        help="Column name for SMILES in the input.",
    )
    parser.add_argument(
        "--label-col", default="active",
        help="Column name for binary 0/1 activity label.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"INFO: {args.input.name} not found; fetching from ChEMBL...",
              file=sys.stderr)
        sys.path.insert(0, str(Path(__file__).parent))
        import download_chembl_coadd

        rc = download_chembl_coadd.main(["--out-dir", str(args.input.parent)])
        if rc != 0:
            print("ERROR: ChEMBL download step failed.", file=sys.stderr)
            return rc
        if not args.input.exists():
            print(f"ERROR: download completed but {args.input} still missing.",
                  file=sys.stderr)
            return 1

    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns or args.label_col not in df.columns:
        print(f"ERROR: required columns missing. Have: {list(df.columns)}", file=sys.stderr)
        return 2

    n_in = len(df)
    df = df.dropna(subset=[args.smiles_col, args.label_col]).copy()
    df["canonical_smiles"] = df[args.smiles_col].astype(str).map(canonicalize)
    df = df.dropna(subset=["canonical_smiles"]).copy()
    df["active"] = df[args.label_col].astype(int).clip(0, 1)
    df = df[["canonical_smiles", "active"]].drop_duplicates(
        subset="canonical_smiles"
    ).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    n_out = len(df)
    n_active = int(df["active"].sum())
    print(f"input   : {args.input}  ({n_in} rows)")
    print(f"output  : {args.output}  ({n_out} rows, {n_active} active = {n_active/n_out:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
