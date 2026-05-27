# Data — CO-ADD P. aeruginosa public dataset

This tutorial uses the **CO-ADD P. aeruginosa phenotypic screening dataset
as mirrored in ChEMBL** (`src_id=40`, `src_short_name=COADD`). CO-ADD
(Community for Open Antimicrobial Drug Discovery, University of Queensland)
releases its screens publicly on ChEMBL after a 24-month embargo.

The actual download path used by this tutorial is the **public ChEMBL REST
API** and **requires no registration**. (Earlier wording in this file asked
you to register at co-add.org — that path exists but is unnecessary for this
tutorial.)

## Reproducing the dataset — one command

```bash
python scripts/prepare_data.py
```

`prepare_data.py` will:

1. Check for `data/raw/coadd_pa_combined_per_molecule.csv`.
2. If missing, automatically call `scripts/download_chembl_coadd.py` to fetch
   the five CO-ADD PA assays from ChEMBL and produce the four lineage CSVs
   in `data/raw/` (~5 minutes, ~25 MB).
3. Canonicalize SMILES, de-duplicate, and write `data/processed/coadd_pa.csv`
   (the ML-ready file).

To force a re-download (e.g. ChEMBL has updated the assays):

```bash
python scripts/download_chembl_coadd.py --force
python scripts/prepare_data.py
```

## Source assays

| ChEMBL assay ID | Type | Description |
|---|---|---|
| `CHEMBL3832903` | Inhibition % | ATCC 27853, 32 µg/mL screen (batch 1) |
| `CHEMBL3832910` | — | empty in source release (skipped) |
| `CHEMBL3832917` | MIC µg/mL | batch 3 |
| `CHEMBL4296187` | Inhibition % | ATCC 27853, 32 µg/mL screen (batch 4) |
| `CHEMBL4296802` | MIC nM | batch 5 |

Total: ~42,636 activity records → ~24,120 unique molecules after per-molecule
aggregation, of which 689 (2.86%) are `active=1`.

## Pipeline

```
ChEMBL REST API  (5 assays)
   │  scripts/download_chembl_coadd.py
   ▼
data/raw/coadd_pseudomonas_aeruginosa_all.csv      ← all activities   (~42,636)
   │
   ├─→ coadd_pa_inhibition_classified.csv          ← inh + per-row active  (~40,919)
   │
   ├─→ coadd_pa_mic.csv                            ← MIC + per-row active  (~1,639)
   │
   └─→ coadd_pa_combined_per_molecule.csv          ← per-molecule, ML-ready (~24,120)
         │  scripts/prepare_data.py  (canonicalize + dedup)
         ▼
   data/processed/coadd_pa.csv                     ← canonical_smiles, active
```

## Active-call rule

A molecule is `active=1` if **any** of the following hold (aggregated across
all assays for that molecule):

- `max_inhibition_pct ≥ 50`
- `min_mic_ugmL ≤ 16`
- `min_mic_nM ≤ 40_000`

A looser `active_30pct` column is written in `coadd_pa_inhibition_classified.csv`
for researchers who want to study the 30–50% partial-activity band.

**Caveat (documented):** the per-molecule rule above is computed from
aggregated `min_mic_*` values **without** rechecking the original
`standard_relation`. A row originally recorded as `MIC > 32 µg/mL` can still
contribute to a `min_mic_nM ≤ 40_000` call after aggregation. This matches
the parent project's combined CSV exactly (24,120 molecules, 689 active) and
is preserved for reproducibility. See `docs/BACKGROUND.md` §5.4 for why this
matters and how to tighten the rule if you need a more conservative count.

## Expected schema after preprocessing

| column | dtype | description |
|---|---|---|
| `canonical_smiles` | str | RDKit-canonical SMILES (de-duplicated) |
| `active` | int | 0 = inactive, 1 = active |

## Sanity check

```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/coadd_pa.csv'); print(df.shape, df['active'].mean())"
# expect:  (24120, 2)  ≈ 0.0286
```

## License & attribution

CO-ADD screening data is © the CO-ADD consortium, distributed via ChEMBL
under their academic-use terms. Cite:

> Blaskovich M.A.T. et al. *Helping chemists discover new antibiotics.*
> ACS Infect. Dis. (2015). CO-ADD project: <https://www.co-add.org>.
>
> Mendez D. et al. *ChEMBL: towards direct deposition of bioassay data.*
> Nucleic Acids Research 47, D930–D940 (2019).
