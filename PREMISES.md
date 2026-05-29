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

**Status**: FALSIFIED on both ECFP4 and Uni-Mol
**Measurement (2026-05-28/29, ECFP4 SAE latent=2048 epochs=30; Uni-Mol SAE latent=4096 epochs=100 on RTX A6000)**:
| Dataset | feature | latent | top10 scaffold% | SAE R²_median |
|---------|---------|--------|-----------------|---------------|
| CO-ADD PA | ECFP4 | 2048 | 16.5% | 0.887 |
| BRAF | ECFP4 | 2048 | 8.5% | 0.872 |
| EGFR | ECFP4 | 2048 | (n/a) | 0.885 |
| JAK2 | ECFP4 | 2048 | (n/a) | 0.918 |
| **BRAF** | **Uni-Mol 512-d** | **4096** | 8.5% | **0.852** |

**Interpretation**: Uni-Mol SAE R² on SAR-narrow BRAF (0.852) is
essentially the same as ECFP4 (0.872) — H2 prediction of R² ≤ 0.60 is
not borne out. H2 FALSIFIED on the dense-embedding case as well.

Per-descriptor R² shows the SAE captures physicochemical descriptors
extremely well (MolWt 0.901, HeavyAtomCount 0.912, TPSA 0.882) and
ring-count descriptors moderately (NumAromaticRings 0.619, RingCount
0.594). The dense Uni-Mol embedding contains the same linear-decodable
descriptor structure as ECFP4 bits — neither feature space loses the
"narrow vs diverse" signal that H2 predicted.

**Open question** (separate from H2): the SAE is *interpretable* for
descriptor recovery but the *number of monosemantic features* (vs
polysemantic) was NOT measured here. The Bharadwaj 2024 paper reports
that as the load-bearing SAE quality metric. R² of RDKit descriptors
is a coarser proxy. Future ticket: measure feature monosemanticity.
**Reference**: Bharadwaj et al. 2024 (arXiv:2512.08077) — SMI-TED SAE on
diverse PubChem reports high variance explained, but no diversity
ablation.
**Owner**: `sae.descriptor_recovery`, T8 in `EXPECTED_OUTPUTS.md`.

### H3 — External BindingDB hold-out exposes covariate shift on kinase

**Claim**: For BRAF, a held-out BindingDB-only set (training-set overlap
removed) will show AUC drop ≥ 0.05 vs. OOF AUC. This is the practical
covariate-shift test the pipeline currently lacks.

**Status**: PARTIALLY VALIDATED — target-dependent (resolved 2026-05-29, GH #3)
**Measurement (ECFP4 scaffold-fold, BindingDB external, overlap removed)**:

| Target | OOF AUC | External AUC | gap | external N (active%) |
|--------|---------|--------------|-----|----------------------|
| BRAF | 0.909 | 0.800 | **+0.109** | 435 (41.4%) |
| EGFR | 0.870 | 0.818 | +0.052 | 1235 (60.1%) |
| JAK2 | 0.921 | 0.981 | **−0.060** | 900 (14.3%) |

**Interpretation**: H3 is **target-dependent, not universal**. BRAF shows
a material covariate shift (0.109); EGFR a mild one (0.052, right at the
threshold); JAK2 shows the external set is *easier* (gap −0.060). The
JAK2 reversal is an artifact of class balance — its external set is only
14.3% active, so AUC is inflated by the easy-negative majority. The
honest claim is **"external validation is necessary because the gap is
unpredictable across targets"** — not "external always drops." For BRAF
the original ≥0.05-drop hypothesis holds; generalizing it to all kinases
is falsified by JAK2.
**Owner**: `examples/run_full_pipeline.py --external`, T3, GH #3.
**dependent_workflows**: external eval is the credibility gate for any
"this pipeline generalizes" claim — and the gap must be reported
per-target, not assumed.

### H4 — MMP rules from kinase data are predominantly series-local

**Claim**: With `SERIES_LOCAL_SCAFFOLD_THRESHOLD = 3`, ≥ 60% of MMP
rules discovered on BRAF will be flagged `is_series_local=True`. On
CO-ADD PA, ≤ 30% should be series-local.

**Status**: VALIDATING (resolved 2026-05-29 via MCS keys — see status update below; earlier FALSIFIED verdict was a key-granularity artifact)
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
this transformation set.

**Status update (2026-05-29 — GH issue #4 resolved): VALIDATING.**
The earlier FALSIFIED verdict was an artifact of transformation-key
granularity, not a property of the data. Three measurements:

| Key encoding | n transformations | series-local |
|--------------|-------------------|--------------|
| Preset 8 SMIRKS | 8 | 12.5% |
| Data-mined, atom-count-delta | 8 | 12.5% |
| **Data-mined, MCS substituent** (`--key-mode mcs`) | **56** | **58.9%** |

With proper mmpdb-grade MCS substituent keys (RDKit `rdFMCS`), 33 of
56 transformations are series-local (< 3 distinct scaffolds) — at the
predicted ≥ 60% threshold (58.9% ≈ 60%). **H4 is supported once the
transformation key is fine enough to resolve individual substituents.**
This is exactly Auer et al. 2016's point: scaffold-locality is visible
only at substituent granularity. The coarse atom-count-delta key
collapses distinct substituent swaps (Cl→F, C→CF3, etc.) into one
bucket, washing out the scaffold-locality signal.

**Side findings (MCS keys)**: BRAF shows interpretable SAR —
`C>>H` (de-methylation) mean Δactive +0.24 across 21 scaffolds (broad,
removing a methyl tends to help); `Cl>>CF3` and `CH2O>>CF3` swaps mean
Δ = −1.00 (strongly hurt, but each only on 6-7 scaffolds = series-local).

**Reference**: Auer et al. 2016 (PMC5198793) — MMP rules are
context-dependent across scaffolds, measured at MCS substituent level.
Our MCS result reproduces this; our coarse-key result was the
methodological null.
**Owner**: `counterfactual.scan` + `scripts/mmp_mine.py --key-mode mcs`, T7, GH #4.

### H6 — Uni-Mol 3D embedding outperforms ECFP4 on cancer-target QSAR

**Claim**: For SAR-dense single-target kinase datasets (BRAF), Uni-Mol
3D embeddings should give higher OOF AUC than ECFP4 bits — that is the
implicit framing of the "3D foundation model > 2D fingerprint"
literature thread the tutorial leans on.

**Status**: FALSIFIED (BRAF only)
**Measurement (2026-05-29, scaffold split, sae_latent=4096)**:
- BRAF Uni-Mol scaffold OOF AUC = **0.804** (AUPRC 0.718)
- BRAF ECFP4 scaffold OOF AUC = **0.909** (AUPRC 0.878)
- gap = **−0.105** (Uni-Mol *loses*)
**Interpretation**: ECFP4 outperforms Uni-Mol on BRAF by a large
margin. Likely reasons: (1) BRAF SAR is dominated by 2D substructural
patterns (specific aromatic substituents, kinase-hinge H-bond donors)
that ECFP4 captures directly; (2) Uni-Mol's pretraining corpus
emphasizes general chemistry and may average out target-specific
discrimination; (3) Uni-Mol's 3D conformer generation introduces
features (conformer ensemble) that don't help for ATP-pocket binders
where 2D pharmacophore is sufficient.

This was the *opposite* of the assumption underlying the tutorial's
choice of Uni-Mol. The README states "Uni-Mol on full N beats ECFP4
here" — that was measured on CO-ADD PA (phenotypic, diverse). On a
target-specific dataset the relationship reverses.
**Caveat**: only measured on BRAF. CO-ADD reproduction with Uni-Mol
(to confirm original 0.895 claim) and EGFR/JAK2 Uni-Mol runs would
strengthen the finding. Both blocked by GPU budget cap (this run hit
$0.19 of approved $0.25).
**Owner**: `examples/run_full_pipeline.py`, README "What this teaches".

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
