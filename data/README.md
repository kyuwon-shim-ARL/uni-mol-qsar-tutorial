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

### Why the censored-MIC caveat matters — and what to do about it

MIC (Minimum Inhibitory Concentration) assays test compounds across a
finite concentration range (CO-ADD PA tops out at 32 µg/mL). When a
compound fails to inhibit growth at the highest tested concentration the
ChEMBL row carries:

| `standard_relation` | `standard_value` | meaning |
|---|---|---|
| `=` | 8 | true MIC measured = 8 µg/mL → genuinely active |
| `>` | 32 | only know MIC > 32 µg/mL; true MIC could be 64, 1000, ∞ → **right-censored = effectively inactive** |

In the CO-ADD MIC table **1,520 of 1,639 rows (93%) are right-censored**
(`relation = ">"`). The per-molecule rule used above ignores the `>` flag
once aggregated, so a `>` row whose value happens to sit at or below the
threshold (e.g. 32 µg/mL ≈ 80,000 nM; CHEMBL4296802's nM batch reports
boundary values in the 40,000-nM range) ends up counted as active.

Reverse-engineered impact on `coadd_pa_combined_per_molecule.csv`:

| label mode | active count | active rate | how computed |
|---|---|---|---|
| **Loose** (shipped default) | 689 | 2.86% | per-molecule rule above, censored rows included |
| **Strict** (censored excluded) | ~89 | 0.37% | additionally require `standard_relation != ">"` on every row that contributes to `min_mic_*` |

The ~600 difference is the false-positive band created by censoring.

#### Loose vs strict — practical trade-off

| | Loose (689 active) | Strict (~89 active) |
|---|---|---|
| Headline OOF AUC | high (~0.83–0.90) | lower (~0.70 ± 0.05 expected) |
| What the model learns | partly "compound was put through MIC assay" (assay-selection bias) | actual "MIC ≤ 32 µg/mL pattern" |
| Transferability to a new dataset | poor — bias differs | better — real activity signal |
| Per-fold validation actives | ~138 | ~18 (workable but noisy) |
| Right comparison metric | AUC | **AUPRC + EF@1%** (AUC saturates at rare-event rate) |

Reference points for the 0.37% active rate: Liu 2023 Gram-negative had
~1.3% actives; Stokes 2020 *halicin* had ~5%; Wong 2024 *abaucin* had ~6.4%.
At 0.37% this tutorial is harder than published successes but still in the
"rare-event learning" regime — not impossible, just demands proper metric
choice and class weighting.

#### How to switch to strict labels

A future-PR sketch (not yet implemented):

```bash
# 1. Refetch with strict-MIC flag (proposed extension; see BACKGROUND.md §5.4 L3):
python scripts/download_chembl_coadd.py --strict-mic --force

# 2. Re-canonicalize / dedup:
python scripts/prepare_data.py
```

Until that flag exists, the same effect can be reproduced by filtering
`coadd_pa_mic.csv` to `standard_relation != ">"` before the per-molecule
combine step.

### Cleanlab — a different angle on noisy labels

Censored MIC is one source of label noise; experimental noise on the
inhibition screen is another. The tutorial ships an **active-protected
Cleanlab cleaner** (`src/qsar_tutorial/label_cleaning.py`) — the binary
analogue of the parent project's P-protected 4-class recipe:

1. 5-fold OOF probabilities on the current feature matrix.
2. `cleanlab.filter.find_label_issues` flags suspicious rows.
3. Flagged `inactive` rows → relabel to argmax (rescuing missed actives).
4. Flagged `active` rows → **protected, never relabeled** (minority class
   is too valuable to wash into the majority).

Enable it in the pipeline:

```bash
python examples/run_full_pipeline.py --features ecfp4 --cleanlab
```

On the N=3,000 smoke run this typically flips ~45 inactive labels to
active and protects ~43 flagged active labels, raising the active count
from 87 to 132 (+45) — a controlled, audit-trailed "minority-class
expansion" that is orthogonal to the strict-MIC fix above.

**What Cleanlab can and cannot do here:**

| | strict-MIC | Cleanlab (active-protected) |
|---|---|---|
| Removes censored false-positive actives | **yes** | no (protected by design) |
| Recovers missed-active (false-negative inactives) | no | **yes** |
| Needs a model | no (rule-based) | yes (uses the pipeline's classifier) |
| Output label set vs default 689 | smaller (~89) | larger (varies, typically 700–800) |

They address different failure modes and can be combined: run strict-MIC
first to drop censored false positives, then Cleanlab to rescue missed
actives among the remaining inactives.

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
