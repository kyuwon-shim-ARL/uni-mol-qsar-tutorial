"""Prepare the CO-ADD P. aeruginosa public dataset for the tutorial.

CO-ADD raw files require registration at:
    https://www.co-add.org

After registration, download the *P. aeruginosa* phenotypic screening data
and place the CSV(s) under  data/raw/  before running this script.

Output:
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
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        print("Download CO-ADD PA data from https://www.co-add.org and place it under data/raw/.",
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
