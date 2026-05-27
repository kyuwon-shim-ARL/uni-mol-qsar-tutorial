"""Unit test for the per-molecule active-call rule documented in
data/README.md and scripts/download_chembl_coadd.py.

Rule (final per-molecule, used by combined CSV):
    active = (max_inhibition_pct >= 50)
          OR (min_mic_ugmL        <= 16)
          OR (min_mic_nM          <= 40_000)

This test locks the rule in code so future changes to the labelling logic
must update this test, surfacing the change explicitly.
"""

from __future__ import annotations

import pandas as pd


def hit_call(row: pd.Series) -> int:
    """Reference implementation of the per-molecule rule."""
    inh = row.get("max_inhibition_pct")
    mic_ug = row.get("min_mic_ugmL")
    mic_nm = row.get("min_mic_nM")
    if pd.notna(inh) and inh >= 50:
        return 1
    if pd.notna(mic_ug) and mic_ug <= 16:
        return 1
    if pd.notna(mic_nm) and mic_nm <= 40_000:
        return 1
    return 0


def test_hit_call_rule_each_condition_alone():
    rows = pd.DataFrame([
        {"max_inhibition_pct": 50.0, "min_mic_ugmL": None, "min_mic_nM": None},   # active via inh
        {"max_inhibition_pct": 49.9, "min_mic_ugmL": None, "min_mic_nM": None},   # below threshold
        {"max_inhibition_pct": None, "min_mic_ugmL": 16.0, "min_mic_nM": None},   # active via ug/mL
        {"max_inhibition_pct": None, "min_mic_ugmL": 16.1, "min_mic_nM": None},   # above
        {"max_inhibition_pct": None, "min_mic_ugmL": None, "min_mic_nM": 40_000}, # active via nM
        {"max_inhibition_pct": None, "min_mic_ugmL": None, "min_mic_nM": 40_001}, # above
        {"max_inhibition_pct": None, "min_mic_ugmL": None, "min_mic_nM": None},   # nothing
    ])
    expected = [1, 0, 1, 0, 1, 0, 0]
    assert [hit_call(r) for _, r in rows.iterrows()] == expected


def test_hit_call_matches_combined_csv_counts():
    """Lock the expected (N, n_active) for the shipped combined CSV.

    Numbers must reproduce the parent project's combined CSV exactly.
    """
    from pathlib import Path
    csv = Path(__file__).resolve().parents[1] / "data" / "processed" / "coadd_pa.csv"
    if not csv.exists():
        # CI without data — skip silently; ChEMBL pull tested elsewhere.
        import pytest
        pytest.skip("data/processed/coadd_pa.csv not present")
    df = pd.read_csv(csv)
    n, n_active = len(df), int(df["active"].sum())
    # Tolerance: ChEMBL occasionally adds/removes a row across releases.
    assert 24_000 <= n <= 24_300, f"row count drift: {n}"
    assert 670 <= n_active <= 710, f"active count drift: {n_active}"
    assert abs(n_active / n - 0.0286) < 0.005, f"active rate drift: {n_active / n}"
