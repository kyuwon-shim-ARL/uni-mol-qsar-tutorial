# PREMISES — testable claims for the QSAR tutorial

Each premise is a claim the pipeline implicitly relies on. Status moves
PRELIM → VALIDATING → VALIDATED (or EXPIRED) as evidence accumulates in
`LANDSCAPE.md`. Adding here is cheap; promoting a claim to VALIDATED
requires explicit measurement.

## H-series (research hypotheses)

### H1 — Scaffold split is mandatory when activity-cliff% ≥ 0.10

**Claim**: On datasets with activity-cliff fraction ≥ 0.10 (Tanimoto≥0.85,
opposite label), random/stratified K-fold AUC overstates generalization by
≥ 0.05. Scaffold split (Bemis-Murcko, GroupKFold) is the minimum defense.

**Status**: WEAKLY-VALIDATED on ECFP4 / Uni-Mol PENDING
**Measurement (2026-05-28, BRAF ECFP4)**:
- stratified OOF AUC = 0.928
- scaffold OOF AUC = 0.909
- gap = **0.019** — below the 0.05 falsification threshold but above 0.
**Interpretation**: gap is real but smaller than the literature
expectation. ECFP4 fingerprints may capture series-bound features less
than dense 3D embeddings; Uni-Mol measurement (blocked on GPU stockout
2026-05-29) is the final arbiter. WEAKLY-VALIDATED in the sense that
"scaffold split is the better practice" remains the recommendation, but
"random K-fold catastrophically inflates AUC" is overstated for ECFP4.
**Owner**: `data.scaffold_folds`, T4 in `EXPECTED_OUTPUTS.md`.
**dependent_workflows**: `examples/run_full_pipeline.py`.

### H2 — SAE descriptor recovery R² depends on chemical diversity

**Claim**: Sparse autoencoder descriptor-recovery R² is upper-bounded by
the chemical diversity of the input distribution. SAR-narrow target
datasets (top-10 scaffold share ≥ 0.30) should not be expected to match
the 0.824 R² achieved on CO-ADD PA (diverse phenotypic).

**Status**: ECFP4-FALSIFIED / Uni-Mol PENDING
**Measurement (2026-05-28/29, ECFP4 SAE latent=2048 epochs=30)**:
| Dataset | top10 scaffold% | SAE R²_median |
|---------|-----------------|---------------|
| CO-ADD PA | 16.5% | 0.887 |
| BRAF | 8.5% | 0.872 |
| EGFR | (n/a measured) | 0.885 |
| JAK2 | (n/a measured) | 0.918 |
**Interpretation**: on ECFP4 (sparse bit vectors), SAE R² is *not*
meaningfully lower for SAR-narrow data. Hypothesis falsified for ECFP4
features. **However the original claim references foundation-model
embeddings** (Bharadwaj et al. used SMI-TED, a dense LM). The ECFP4 path
may be uninformative for the H2 question — sparse bit vectors are
already over-specified and easy to compress regardless of input
diversity. **Uni-Mol SAE measurement is the load-bearing test** and
remains blocked on GPU stockout (2026-05-29).
**Reference**: Bharadwaj et al. 2024 (arXiv:2512.08077) — SMI-TED SAE on
diverse PubChem reports high variance explained, but no diversity
ablation.
**Owner**: `sae.descriptor_recovery`, T8 in `EXPECTED_OUTPUTS.md`.

### H3 — External BindingDB hold-out exposes covariate shift on kinase

**Claim**: For BRAF, a held-out BindingDB-only set (training-set overlap
removed) will show AUC drop ≥ 0.05 vs. OOF AUC. This is the practical
covariate-shift test the pipeline currently lacks.

**Status**: VALIDATED on BRAF / cross-kinase pending
**Measurement (2026-05-28, BRAF ECFP4 scaffold-fold)**:
- OOF AUC = 0.909
- BindingDB external AUC = 0.800
- gap = **0.109** (above 0.05 falsification threshold)
**Interpretation**: confirmed. External drop is real and material. The
0.10 gap is large enough that any "the pipeline generalizes" claim
without external validation should be challenged. EGFR / JAK2 external
BindingDB measurements are not yet performed (pending separate fetch).
**Owner**: `examples/run_full_pipeline.py --external`, T3.
**dependent_workflows**: external eval is the credibility gate for any
"this pipeline generalizes" claim.

### H4 — MMP rules from kinase data are predominantly series-local

**Claim**: With `SERIES_LOCAL_SCAFFOLD_THRESHOLD = 3`, ≥ 60% of MMP
rules discovered on BRAF will be flagged `is_series_local=True`. On
CO-ADD PA, ≤ 30% should be series-local.

**Status**: FALSIFIED for the default transformation set
**Measurement (2026-05-28, BRAF MMP grid ablation, ECFP4 model)**:
| Inactive subset | threshold=2 | threshold=3 | threshold=5 |
|-----------------|-------------|-------------|-------------|
| 500 | 0/8 (0%) | 0/8 (0%) | 0/8 (0%) |
| 1500 | 0/8 (0%) | 0/8 (0%) | 0/8 (0%) |
| 3401 (full) | 0/8 (0%) | 0/8 (0%) | 0/8 (0%) |

| Target | series-local fraction (threshold=3) |
|--------|-------------------------------------|
| CO-ADD PA | 0/8 (0%) |
| BRAF | 1/8 (12.5%) (default 500 subset) |
| EGFR | 0/8 (0%) |
| JAK2 | 1/8 (12.5%) |

**Interpretation**: The default 8-SMIRKS transformation set
(`add_F_aromatic`, `add_OH_aromatic`, etc.) applies *broadly* across
all observed scaffolds — every rule reaches > 5 distinct scaffolds. The
hypothesis (kinase MMP rules are scaffold-bound) cannot be tested with
this transformation set. **Data-mined MMP test (2026-05-29,
`scripts/mmp_mine.py` on BRAF)**: with Tanimoto ≥ 0.85 + same scaffold
filter + atom-count-delta bucketing, 8 transformation buckets were
discovered (delta_{-4,-3,-2,-1,+0,+1,+2,+3}). Series-local fraction =
1/8 = **12.5%** — same magnitude as the default-SMIRKS result. So the
data-mining approach (with coarse delta-atom-count keys) does not
expose a different pattern.

The transformation *granularity* matters more than data-mined vs
preset choice. A finer key (MCS-based substituent identity) would
split these buckets into many more transformations and might expose
more series-local rules — but `mmpdb`-grade implementation is out of
the tutorial scope.

**Side finding**: BRAF data-mined MMPs reveal a clear "smaller is
more potent" pattern — delta_-3 (102 fewer atoms across pairs) shows
100% probability of the smaller compound being active, delta_+3
shows 3.3%. Consistent with BRAF ATP pocket geometric constraints.

**Reference**: Auer et al. 2016 (PMC5198793) — MMP rules are
context-dependent across scaffolds. (Their finding is about
data-mined MMPs at MCS-level granularity — consistent with our
"granularity matters" conclusion.)
**Owner**: `counterfactual.scan` + `scripts/mmp_mine.py`, T7.

### H5 — Kinase QSAR has a 2014/2015 temporal break

**Claim**: Models trained on BRAF compounds deposited ≤ 2014 do not
predict ≥ 2015 compounds well. Models trained ≥ 2016 predict well
within their era. There is a chemical-space discontinuity around the
2014/2015 boundary, likely driven by the introduction of new BRAF
inhibitor scaffolds in the post-vemurafenib era.

**Status**: VALIDATING (BRAF only, ECFP4)
**Measurement (2026-05-29, time-sweep ecfp4)**:
| Cutoff | n_train | n_test | AUC |
|--------|---------|--------|-----|
| 2010 | 998 | 4753 | 0.548 |
| 2012 | 1572 | 4179 | 0.569 |
| 2014 | 2783 | 2968 | 0.603 |
| 2016 | 3647 | 2104 | 0.859 |
| 2018 | 5092 | 659 | 0.730 |
| 2020 | 5448 | 303 | 0.875 |

**Interpretation**: AUC jumps from 0.60 (cutoff 2014) to 0.86 (cutoff
2016). The 2018 dip is likely test-set size noise (n=659). The cleanest
interpretation: pre-2014 training set has missing scaffolds that appear
post-2014 — the model literally cannot have seen the chemistry it is
being asked to predict. Implication: **time-split is the most honest
single-number metric** for kinase QSAR. OOF AUC silently averages over
this temporal break.
**Owner**: `data.time_split_indices`, T5; `scripts/time_split_sweep.py`.
**dependent_workflows**: any "predict future BRAF inhibitor" claim.

## P-series (procedural premises)

### P1 — pChEMBL ≥ 8 is the working active cutoff for ChEMBL kinase data

**Claim**: pChEMBL ≥ 8.0 (IC50 ≤ 10 nM) gives a 30–50% active fraction
on ChEMBL kinase targets, suitable for binary classification. Lower
cutoffs (≥ 6.0 = 1 μM) yield 80–90% active due to publication bias.

**Status**: VALIDATED for BRAF (measured 40.9% active at cutoff=8.0;
87.4% at cutoff=6.0). Generalization to other kinases unmeasured.
**Owner**: `scripts/download_chembl_target.py --pchembl-cutoff`.

### P2 — Diagnose before train

**Claim**: Any new target dataset must pass `scripts/diagnose_dataset.py`
before downstream interpretation is treated as credible. Specifically,
the cliff fraction determines the required split mode (random vs
scaffold), and the SAE R² ceiling depends on top-10 scaffold share.

**Status**: VALIDATED (institutional policy as of 2026-05-28; see
`EXPECTED_OUTPUTS.md` "Mandatory practice for target-specific runs").
**Owner**: `scripts/diagnose_dataset.py`, T1/T2.

---

When a premise is falsified, move it to a `## DEPRECATED` section at the
bottom rather than deleting — the lineage is informative.
