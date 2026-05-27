#!/usr/bin/env python3
"""Download CO-ADD P. aeruginosa screening data from ChEMBL.

The tutorial's "CO-ADD" dataset is hosted in ChEMBL (source ID 40,
src_short_name=COADD). It is *not* downloaded directly from co-add.org
and no registration is required to obtain the activity data this tutorial
uses. CO-ADD releases its screens on ChEMBL after a 24-month embargo.

This script:
  1. Queries the ChEMBL REST API for five CO-ADD PA assays.
  2. Writes the raw activities CSV.
  3. Splits and classifies the inhibition and MIC subsets.
  4. Aggregates per molecule into the ML-ready combined CSV.

Source assays (ChEMBL src_id=40):
    CHEMBL3832903   inhibition %   ATCC 27853, 32 ug/mL screen (batch 1)
    CHEMBL3832910   empty          skipped (zero records in ChEMBL)
    CHEMBL3832917   MIC ug/mL      batch 3
    CHEMBL4296187   inhibition %   ATCC 27853, 32 ug/mL screen (batch 4)
    CHEMBL4296802   MIC nM         batch 5

Active-call rules used in this script:

    Intermediate (per-row, for documentation only):
        inhibition_active_50pct = standard_value >= 50
        inhibition_active_30pct = standard_value >= 30          (loose)
        mic_active (per row)    = (value <= 16 ug/mL OR <= 40_000 nM)
                                  AND standard_relation != ">"  (not censored)

    Final per-molecule rule (used by combined CSV):
        active = (max_inhibition_pct >= 50)
              OR (min_mic_ugmL        <= 16)
              OR (min_mic_nM          <= 40_000)

    The per-molecule rule deliberately matches the parent project's behavior
    so the 24,120 / 689 active counts reproduce exactly. It does NOT respect
    censored MIC relations (e.g. "MIC > 32 ug/mL" rows can satisfy the
    min_mic_nM <= 40_000 condition once aggregated). See BACKGROUND.md §5.4
    "Known limitations" for why this matters and how to tighten it.

Outputs (written to data/raw/):
    coadd_pseudomonas_aeruginosa_all.csv      ~42,636 rows  (raw activities)
    coadd_pa_inhibition_classified.csv        ~40,919 rows  (inh + active calls)
    coadd_pa_mic.csv                          ~1,639  rows  (MIC + active call)
    coadd_pa_combined_per_molecule.csv        ~24,120 rows  (per-molecule ML-ready)

Network access is required (~5 minutes, ~25 MB on first run). Re-running with
all four output CSVs already present is a no-op unless --force is given.

Usage:
    python scripts/download_chembl_coadd.py
    python scripts/download_chembl_coadd.py --force
    python scripts/download_chembl_coadd.py --out-dir data/raw
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

# CO-ADD PA assays in ChEMBL. CHEMBL3832910 has zero records (placeholder
# in the source release) and is intentionally absent from this map.
ASSAYS: dict[str, dict[str, str]] = {
    "CHEMBL3832903": {"kind": "inhibition", "desc": "ATCC 27853, 32 ug/mL screen (batch 1)"},
    "CHEMBL3832917": {"kind": "mic",        "desc": "MIC batch 3"},
    "CHEMBL4296187": {"kind": "inhibition", "desc": "ATCC 27853, 32 ug/mL screen (batch 4)"},
    "CHEMBL4296802": {"kind": "mic",        "desc": "MIC batch 5"},
}

KEEP_COLS = [
    "molecule_chembl_id", "canonical_smiles",
    "standard_type", "standard_value", "standard_units", "standard_relation",
    "assay_chembl_id", "assay_description",
    "target_organism", "target_pref_name",
    "activity_id", "document_chembl_id", "document_year",
    "data_validity_comment", "potential_duplicate",
]


# ----------------------------- ChEMBL fetch ---------------------------------

def _http_json(url: str, *, timeout: int = 45, retries: int = 4) -> dict:
    """GET a ChEMBL endpoint with exponential-backoff retries."""
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


def _count_records(assay_id: str) -> int:
    params = {"assay_chembl_id": assay_id, "format": "json", "limit": 1, "offset": 0}
    url = f"{CHEMBL_BASE}/activity?" + urllib.parse.urlencode(params)
    return int(_http_json(url, timeout=30)["page_meta"]["total_count"])


def _fetch_page(assay_id: str, offset: int, limit: int = 1000) -> list[dict]:
    params = {"assay_chembl_id": assay_id, "format": "json",
              "limit": limit, "offset": offset}
    url = f"{CHEMBL_BASE}/activity?" + urllib.parse.urlencode(params)
    return _http_json(url).get("activities", [])


def _download_assay(assay_id: str) -> list[dict]:
    total = _count_records(assay_id)
    if total == 0:
        print(f"  {assay_id}: 0 records (skipped)")
        return []
    print(f"  {assay_id}: {total} records  ({ASSAYS[assay_id]['desc']})")
    offsets = list(range(0, total, 1000))
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(_fetch_page, assay_id, off) for off in offsets]
        for fut in as_completed(futures):
            records.extend(fut.result())
    return records


# ----------------------------- Pipeline steps -------------------------------

def step_download(raw_dir: Path) -> Path:
    """Step 1: pull all PA assays from ChEMBL, write raw activities CSV."""
    all_records: list[dict] = []
    for assay_id in ASSAYS:
        all_records.extend(_download_assay(assay_id))
    df = pd.DataFrame(all_records)
    df = df[[c for c in KEEP_COLS if c in df.columns]]
    df = df.dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    out = raw_dir / "coadd_pseudomonas_aeruginosa_all.csv"
    df.to_csv(out, index=False)
    print(f"  -> {out.name}  ({len(df)} rows)")
    return out


def step_classify_inhibition(all_csv: Path, raw_dir: Path) -> Path:
    """Step 2: extract inhibition rows, write active_50pct / active_30pct."""
    df = pd.read_csv(all_csv, low_memory=False)
    inh = df[df["standard_type"] == "Inhibition"].copy()
    val = pd.to_numeric(inh["standard_value"], errors="coerce")
    inh["active_50pct"] = (val >= 50).fillna(False).astype(int)
    inh["active_30pct"] = (val >= 30).fillna(False).astype(int)
    cols = [
        "molecule_chembl_id", "canonical_smiles", "standard_value",
        "active_50pct", "active_30pct",
        "assay_chembl_id", "assay_description",
        "data_validity_comment", "potential_duplicate",
    ]
    inh = inh[[c for c in cols if c in inh.columns]]
    out = raw_dir / "coadd_pa_inhibition_classified.csv"
    inh.to_csv(out, index=False)
    print(f"  -> {out.name}  ({len(inh)} rows, "
          f"{int(inh['active_50pct'].sum())} active@50%)")
    return out


def step_classify_mic(all_csv: Path, raw_dir: Path) -> Path:
    """Step 3: extract MIC rows, harmonize units, write active call."""
    df = pd.read_csv(all_csv, low_memory=False)
    mic = df[df["standard_type"] == "MIC"].copy()
    val = pd.to_numeric(mic["standard_value"], errors="coerce")
    units = mic["standard_units"].fillna("").astype(str)
    rel = mic["standard_relation"].fillna("=").astype(str)

    is_ugml = units.str.contains("ug", case=False, regex=False)
    is_nm = units.str.contains("nM", case=False, regex=False)

    # Active = MIC <= 16 ug/mL OR <= 40_000 nM, AND not right-censored.
    threshold_hit = (is_ugml & (val <= 16)) | (is_nm & (val <= 40_000))
    not_censored = rel != ">"
    mic["active"] = (threshold_hit & not_censored).fillna(False).astype(int)
    mic["mic_ugmL"] = val.where(is_ugml)
    mic["mic_nM"] = val.where(is_nm)
    cols = [
        "molecule_chembl_id", "canonical_smiles", "mic_ugmL", "mic_nM",
        "standard_relation", "active",
        "assay_chembl_id", "assay_description",
    ]
    mic = mic[[c for c in cols if c in mic.columns]]
    out = raw_dir / "coadd_pa_mic.csv"
    mic.to_csv(out, index=False)
    print(f"  -> {out.name}  ({len(mic)} rows, "
          f"{int(mic['active'].sum())} active)")
    return out


def step_combine(inh_csv: Path, mic_csv: Path, raw_dir: Path) -> Path:
    """Step 4: aggregate per molecule. max inhibition, min MIC, OR of actives."""
    inh = pd.read_csv(inh_csv)
    mic = pd.read_csv(mic_csv)

    inh_val = pd.to_numeric(inh["standard_value"], errors="coerce")
    inh_g = (
        inh.assign(_v=inh_val)
           .groupby("molecule_chembl_id", as_index=False)
           .agg(
               canonical_smiles=("canonical_smiles", "first"),
               max_inhibition_pct=("_v", "max"),
               inh_active=("active_50pct", "max"),
           )
    )
    mic_g = (
        mic.groupby("molecule_chembl_id", as_index=False)
           .agg(
               canonical_smiles=("canonical_smiles", "first"),
               min_mic_ugmL=("mic_ugmL", "min"),
               min_mic_nM=("mic_nM", "min"),
               mic_active=("active", "max"),
           )
    )

    combined = pd.merge(
        inh_g, mic_g, on="molecule_chembl_id",
        how="outer", suffixes=("", "_mic"),
    )
    # Inhibition table covers more molecules; consolidate SMILES from MIC
    # for the rest.
    combined["canonical_smiles"] = combined["canonical_smiles"].fillna(
        combined["canonical_smiles_mic"]
    )

    # Per-molecule active rule. Recomputed from aggregated raw values rather
    # than ORing the per-row active flags. This is intentional: it matches the
    # parent project's combined CSV byte-for-byte (24,120 molecules, 689
    # active). It does NOT respect censored MIC relations — a documented
    # limitation kept here for reproducibility (see module docstring + the
    # BACKGROUND.md §5.4 "Known limitations" entry).
    inh_ok = combined["max_inhibition_pct"].fillna(-np.inf) >= 50
    mic_ugml_ok = combined["min_mic_ugmL"].fillna(np.inf) <= 16
    mic_nm_ok = combined["min_mic_nM"].fillna(np.inf) <= 40_000
    combined["active"] = (inh_ok | mic_ugml_ok | mic_nm_ok).astype(int)

    out_cols = [
        "molecule_chembl_id", "max_inhibition_pct",
        "min_mic_ugmL", "min_mic_nM",
        "canonical_smiles", "active",
    ]
    combined = (
        combined[out_cols]
        .dropna(subset=["canonical_smiles"])
        .reset_index(drop=True)
    )
    out = raw_dir / "coadd_pa_combined_per_molecule.csv"
    combined.to_csv(out, index=False)
    print(f"  -> {out.name}  ({len(combined)} molecules, "
          f"{int(combined['active'].sum())} active "
          f"= {combined['active'].mean():.2%})")
    return out


# --------------------------------- CLI --------------------------------------

def expected_outputs(raw_dir: Path) -> dict[str, Path]:
    return {
        "all":      raw_dir / "coadd_pseudomonas_aeruginosa_all.csv",
        "inh":      raw_dir / "coadd_pa_inhibition_classified.csv",
        "mic":      raw_dir / "coadd_pa_mic.csv",
        "combined": raw_dir / "coadd_pa_combined_per_molecule.csv",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
        help="Directory to write the four lineage CSVs (default: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if all outputs already exist.",
    )
    args = parser.parse_args(argv)

    raw_dir = args.out_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets = expected_outputs(raw_dir)

    if not args.force and all(p.exists() for p in targets.values()):
        print(f"All raw CSVs already present in {raw_dir}. Use --force to refetch.")
        return 0

    print(f"[1/4] Downloading PA bioactivities from ChEMBL  (~5 min, ~25 MB)")
    step_download(raw_dir)
    print(f"[2/4] Classifying inhibition screen")
    step_classify_inhibition(targets["all"], raw_dir)
    print(f"[3/4] Classifying MIC screen")
    step_classify_mic(targets["all"], raw_dir)
    print(f"[4/4] Aggregating per molecule")
    step_combine(targets["inh"], targets["mic"], raw_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
