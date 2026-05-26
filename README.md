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
