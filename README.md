# Interpretable QSAR Tutorial — Uni-Mol + XGBoost + SHAP + SAE + MMP

A teaching repository for new interns starting on molecular property
prediction and model interpretation. It runs the full stack from a 3D
foundation-model embedding (Uni-Mol) through a tree-model predictor, three
parallel interpretation layers (SHAP, Sparse Autoencoder, matched molecular
pair counterfactual), and a self-contained HTML report — all on the public
**CO-ADD** *P. aeruginosa* dataset.

> **No internal data is included.** All scripts are written against the
> public CO-ADD PA binary (active/inactive) screen. You download CO-ADD
> yourself; see `data/README.md`.

---

## What this teaches

1. **Why each layer exists.** Read `docs/BACKGROUND.md` first — it places
   every tool in this repo on a generational map (Hansch 1964 → Lipinski
   1997 → ECFP4+RF → GNNs → Uni-Mol → SAE) and shows what problem each
   generation was invented to solve.
0. **What numbers to expect.** `docs/EXPECTED_OUTPUTS.md` records the
   real metrics this pipeline produced on an NVIDIA A40 with the full
   CO-ADD PA dataset (AUC=0.895, SAE descriptor R²=0.824, 8 MMP rules
   with bootstrap CIs). Use it to tell if your setup is broken.
2. **How to assemble the stack.** Six small modules, each ≤200 lines.
3. **How to report results honestly.** L1 (predictive) + L2 (mechanistic
   hypothesis) — and no further. The report template enforces caveats.

## Pipeline at a glance

```
SMILES (CO-ADD PA)
   │
   ▼  featurizer.py
Uni-Mol embedding  (or ECFP4 baseline)
   │
   ▼  model.py
XGBoost binary classifier  ──►  OOF AUC / AUPRC / MCC
   │
   ├──►  shap_layer.py        TreeSHAP per-feature attribution
   │
   ├──►  sae.py               Sparse Autoencoder → descriptor recovery
   │
   └──►  counterfactual.py    MMP transformations → Δp(active)
                                  │
                                  ▼  report.py
                              Self-contained HTML report
```

## Quickstart

```bash
# 1. install
python -m pip install -e ".[notebooks]"
pip install unimol-tools          # heavy: torch + HF download; install separately

# 2. data
#    follow data/README.md to download CO-ADD PA, then:
python scripts/prepare_data.py

# 3. run end-to-end example
python examples/run_full_pipeline.py
```

The example writes `reports/coadd_pa_report.html`.

## Repo layout

```
uni-mol-qsar-tutorial/
├── README.md                    you are here
├── docs/
│   └── BACKGROUND.md            ← READ FIRST — the field landscape
├── data/
│   └── README.md                how to obtain CO-ADD
├── scripts/
│   └── prepare_data.py          raw CO-ADD CSV → canonicalized CSV
├── src/qsar_tutorial/
│   ├── data.py                  load + stratified split + K-fold
│   ├── featurizer.py            Uni-Mol + ECFP4 baseline
│   ├── model.py                 XGBoost + OOF protocol
│   ├── shap_layer.py            TreeSHAP wrapper
│   ├── sae.py                   Sparse Autoencoder + diagnostics
│   ├── counterfactual.py        MMP scanning + bootstrap CI
│   └── report.py                Jinja2 HTML report
├── examples/
│   └── run_full_pipeline.py     ties everything together
├── notebooks/
│   ├── 01_data_and_baseline.ipynb
│   ├── 02_unimol_xgb_shap.ipynb
│   └── 03_sae_counterfactual_report.ipynb
└── tests/
    └── test_smoke.py            import + tiny-data sanity
```

## Honest framing — please read

This tutorial reproduces the *interpretability* stack of a larger internal
research project. The internal data, labels, and downstream wet-lab
validation are not in this repo. Specifically:

- The original project uses a 4-class mechanistic label (W / E / P / EP).
  CO-ADD is binary (active / inactive). You will see the same pipeline
  produce useful results, but the *biological resolution* is lower.
- The original project's headline design rule (propargylamine substitution)
  was discovered on the 4-class dataset and only counts as
  *hypothesis-generating* without wet-lab confirmation. On CO-ADD binary,
  treat any rule from `counterfactual.py` as an even earlier-stage
  hypothesis.
- Read `docs/BACKGROUND.md` §5 for the L1–L4 evidence pyramid that frames
  what claims this tutorial can and cannot support.

## Cancer-target extension (BRAF and friends)

The same pipeline works on any SMILES + binary label CSV. As of 2026-05-28
the repo includes an extension for single-target cancer-protein QSAR
(BRAF, EGFR, JAK2 demonstrated). Key entry points:

```bash
# 1. Pull a ChEMBL target → SMILES + active (binary at pchembl ≥ 8)
python scripts/download_chembl_target.py --target CHEMBL5145 --name braf
python scripts/prepare_data.py --input data/raw/braf_per_molecule.csv \
                               --output data/processed/braf.csv

# 2. Diagnose suitability BEFORE training (read the verdict)
python scripts/diagnose_dataset.py data/processed/braf.csv

# 3. Run pipeline with auto-split (scaffold for SAR-dense data)
PYTHONPATH=src python examples/run_full_pipeline.py \
    --csv data/processed/braf.csv --features ecfp4 \
    --time-cutoff 2015 \
    --external data/external/braf_bindingdb.csv \
    --out reports/braf_ecfp4.html
```

What's different from the CO-ADD path:
- **`scripts/diagnose_dataset.py`** prints a 4-rule verdict (cliff%,
  scaffold dominance, class balance) — refuse to interpret downstream
  results when cliff% ≥ 0.10 is unhandled.
- **`--split auto`** chooses scaffold-fold for non-CO-ADD CSVs by default
  (CO-ADD path unchanged for byte-compat reproducibility).
- **`--time-cutoff YEAR`** trains on `document_year ≤ cutoff`, evals on
  `> cutoff` — Sheridan 2013-style "past predicts future" hold-out.
- **`--external CSV`** scores an independent held-out set (e.g.
  BindingDB) with auto color-coded gap caveat.
- **MMP rules** are annotated with `n_distinct_scaffolds` and flagged
  `series-local` (3 scaffolds threshold by default; see
  `counterfactual.SERIES_LOCAL_SCAFFOLD_THRESHOLD`).
- **HTML report** auto-includes a scope boilerplate from
  `docs/EXPLAINABILITY_SCOPE.md` when the data is single-target.

### Intern walkthrough checklist

If you are an intern starting on this extension, work through:

1. ☐ Read `docs/EXPLAINABILITY_SCOPE.md` (1 page) — what the pipeline
   answers vs. doesn't.
2. ☐ Read `PREMISES.md` (H1–H4 + P1–P2) — the testable claims this
   project relies on. Each premise has a falsification condition.
3. ☐ Run `python scripts/diagnose_dataset.py data/processed/braf.csv` —
   recognize each row in the output.
4. ☐ Run the example pipeline on BRAF (above commands). Confirm the
   HTML report renders.
5. ☐ Open `reports/20260528_braf-full_ecfp4.json` (or your fresh run).
   Note the OOF AUC vs `time_split.AUC` vs `external.AUC`.
   *The gap between these three is the actual lesson.*
6. ☐ Record in `LANDSCAPE.md` (under "Pending measurements") any
   premise check your run resolved.
7. ☐ Where you got stuck → open an issue or add to the
   `data/external/README.md` "Known external risks" list.

## License

MIT (see `LICENSE`). CO-ADD data follows its own academic-use terms.

## Acknowledgements

Built on top of:
- **Uni-Mol** (Zhou et al. 2023) — 3D molecular foundation model.
- **CO-ADD** (Blaskovich et al., 2015–) — public Gram-negative
  phenotypic data.
- **SHAP** (Lundberg & Lee 2017) and **TreeSHAP** (Lundberg et al. 2020).
- **Sparse Autoencoder for chemistry**: Cohen 2025 (SMI-TED) and the
  general recipe from Bricken et al. 2023.
