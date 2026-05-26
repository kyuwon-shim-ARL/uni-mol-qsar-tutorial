# Data — CO-ADD P. aeruginosa public dataset

This tutorial uses public CO-ADD (Community for Open Antimicrobial Drug
Discovery) phenotypic screening data against *P. aeruginosa*. No internal
or proprietary data is included in this repository.

## Why CO-ADD

- Public, license-friendly for academic research.
- Large (~24K compounds for PA).
- Well curated by the CO-ADD consortium (Blaskovich et al., since 2015).
- Standard external-validation set in the Gram-negative MMPA literature
  (Gurvic 2024 and follow-ups).

## How to get the data

1. Register at <https://www.co-add.org> (free for academic use).
2. Download the *P. aeruginosa* phenotypic screening files.
3. Place the CSV under `data/raw/` (default name expected:
   `coadd_pa_combined_per_molecule.csv`).
4. Run `python scripts/prepare_data.py` to produce
   `data/processed/coadd_pa.csv` with two columns:

   ```
   canonical_smiles , active
   ```

## Expected schema after preprocessing

| column            | dtype  | description |
|-------------------|--------|-------------|
| `canonical_smiles` | str    | RDKit-canonical SMILES (de-duplicated) |
| `active`           | int    | 0 = inactive, 1 = active (PA growth inhibition above cutoff) |

## Sanity checks

After `prepare_data.py`:

```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/coadd_pa.csv'); print(df.shape, df['active'].mean())"
```

Expect roughly N≈24,000 unique molecules with active fraction ≈ 0.03
(few-percent active rate is the usual Gram-negative phenotypic signature).

## License & attribution

CO-ADD data is © the CO-ADD consortium and distributed under their
academic-use terms. Cite:

> Blaskovich M.A.T. et al. *Helping chemists discover new antibiotics.*
> ACS Infect. Dis. (2015) — and the CO-ADD project at <https://www.co-add.org>.
