# LANDSCAPE — evidence registry for the QSAR tutorial

Substrate for evidence that supports, attacks, or qualifies the premises
in `PREMISES.md`. Each entry has: claim, source, polarity, ref. New
entries accumulate; old entries are not deleted (lineage matters).

## Schema

```
- id        : L-{date}-{N}
- type      : premise_evidence | landscape_evidence | open_question
- refers_to : PREMISES.md#H{N} or PREMISES.md#P{N}, or null
- polarity  : supports | attacks | contested | qualifies
- source    : { kind: openalex | external-search | llm-inference | measured | code-audit
              , ref: DOI / URL / file path / commit hash }
- claim     : one sentence
- date      : YYYY-MM-DD
```

## Evidence

### Validated practices

**L-20260528-01** — refers_to: H1 — polarity: supports — source: code-audit
*src/qsar_tutorial/data.py:96-136*. The pipeline ships with
`scaffold_folds(GroupKFold over Bemis-Murcko)` already implemented; the
docstring states "Expect AUC to drop 0.05-0.15 vs the random split."
This is the *prior* — H1 in PREMISES.md is the *measurement* of that
gap on BRAF specifically.

**L-20260528-02** — refers_to: H1 — polarity: supports — source: openalex
*van Tilborg et al. 2022, PMC9749029*. Activity cliffs degrade both
QSAR accuracy and SHAP attributions; this is a property of the data,
not the model. Implication: scaffold split is necessary, not sufficient.

**L-20260528-03** — refers_to: H4 — polarity: supports — source: openalex
*Auer et al. 2016, J. Chem. Inf. Model., PMC5198793*. MMP rule
significance is scaffold-dependent — same transformation has near-zero
mean Δ activity when aggregated across scaffolds. Justifies the
`is_series_local` flag (threshold = 3 distinct Murcko scaffolds).

**L-20260528-04** — refers_to: H4 — polarity: supports — source: openalex
*PMC11032345 (2024)* — Kinase family has documented unusually high
activity-cliff propensity. Sixteen significant BRAF cliff generators
catalogued. Justifies elevated MMP scrutiny on kinase data.

### Open empirical questions

**L-20260528-05** — refers_to: H2 — polarity: contested — source: openalex
*Bharadwaj et al. 2024, arXiv:2512.08077* — SAE on SMI-TED reports
>97% reconstruction R² on diverse PubChem corpus. **Does not report**
diversity ablation. So this paper supports SAE-on-foundation-model
viability in general but neither confirms nor refutes H2. Genuine
gap — H2 measurement on BRAF will be the first datapoint.

**L-20260528-06** — refers_to: H2 — polarity: qualifies — source: openalex
*arXiv:2503.05613 (SAE survey, 2025)* — narrow input distribution
yields fewer monosemantic SAE features. Qualitative statement, no
threshold or formula. Use as theoretical anchor for H2 if measurement
confirms.

**L-20260528-07** — refers_to: H3 — polarity: qualifies — source: openalex
*Cai et al. 2022, PMC9208089* — TreeSHAP on kinase families recovers
ATP-binding hinge pharmacophores validated against PDB. Shows
interpretability *can* be mechanistically meaningful on kinase, but
the paper did not measure covariate shift on external held-out set.

### Diagnostic measurements (2026-05-28)

**L-20260528-08** — refers_to: P1 — polarity: supports — source: measured
*data/raw/braf_per_molecule.csv*. BRAF (CHEMBL5145): 9,834 raw records
→ 5,751 unique molecules. pchembl distribution: median 7.68, std 1.23.
At pchembl ≥ 8.0 cutoff: 40.9% active. At pchembl ≥ 6.0: 87.4%.

**L-20260528-09** — refers_to: H1, P2 — polarity: supports — source: measured
*scripts/diagnose_dataset.py outputs (2026-05-28)*. Activity-cliff
fraction (Tanimoto ≥ 0.85, opposite label):
- CO-ADD PA: 0.1% (n=4 / 6000 subsample)
- BRAF (ChEMBL): 11.1%
- BRAF (BindingDB external): 30.6%

The CO-ADD/BRAF gap is the empirical basis for "phenotypic vs target-
specific" distinction in `EXPECTED_OUTPUTS.md`. The BindingDB external
cliff fraction (3× BRAF training) confirms it is a genuine stress
test, not a redundant sample.

**L-20260528-10** — refers_to: H3 — polarity: supports — source: measured
*data/external/braf_bindingdb.csv*. BindingDB BRAF (P15056): 3,587
unique molecules pre-subtract → 87.9% overlap with ChEMBL set → 435
external-only molecules, 41.4% active. Adequate for AUC estimation on
held-out set (n_active=180).

### Pending measurements

**L-20260528-11** — refers_to: H1 — polarity: supports (weak) — source: measured
*reports/20260528_braf-*_ecfp4.json*. BRAF ECFP4 OOF AUC:
- stratified split = 0.928
- scaffold split = 0.909
- gap = 0.019. Below the 0.05 literature expectation but above 0 —
  scaffold split is the better practice but the catastrophic-leakage
  framing is overstated for ECFP4. Uni-Mol comparison still pending.

**L-20260528-12** — refers_to: H2 — polarity: refutes (ECFP4 only) — source: measured
*All 4 ECFP4 SAE R² measurements (CO-ADD 0.887, BRAF 0.872, EGFR 0.885,
JAK2 0.918)*. SAE R² does NOT meaningfully decrease for SAR-narrow
datasets when features are ECFP4 bits. Either H2 is wrong, or ECFP4 is
the wrong representation for testing it. Bharadwaj 2024 (L-20260528-05)
used dense foundation-model embeddings — likely the load-bearing
representation. Uni-Mol SAE measurement is the discriminating test
(blocked on GPU stockout 2026-05-29).

**L-20260528-13** — refers_to: H3 — polarity: supports — source: measured
*reports/20260528_braf-full_ecfp4.json*. BRAF scaffold OOF AUC 0.909 →
BindingDB external AUC 0.800 = gap 0.109. Above the 0.05 threshold.
Confirms that ChEMBL-only training overestimates real-world
generalization — external set is necessary.

**L-20260528-14** — refers_to: H4 — polarity: refutes (default SMIRKS) — source: measured
*reports/20260529_braf_mmp_grid.json*. All 9 (subset × threshold) cells
on BRAF produce 0/8 series-local rules. Default 8-SMIRKS set is too
generic to produce series-bound rules. To test the Auer 2016 claim
(data-mined MMPs are scaffold-bound) the pipeline would need to mine
MMPs from the data itself, not apply a preset list. Deferred.

**L-20260529-15** — refers_to: H5 — polarity: supports — source: measured
*reports/20260529_braf_time_sweep.json*. BRAF time-split AUC sweep:
cutoff 2010=0.548, 2012=0.569, 2014=0.603, 2016=0.859, 2018=0.730
(noisy), 2020=0.875. **The 2014→2016 jump (0.603 → 0.859) is the
strongest single finding of the extension.** Models trained on
pre-2014 BRAF chemistry literally cannot see post-vemurafenib
scaffolds. Time-split is the most honest single-number metric.

**L-20260529-16** — refers_to: H3 (general) — polarity: supports — source: measured
*Cross-target ECFP4 AUC measurements*. EGFR scaffold AUC=0.870
(N=11185), JAK2 scaffold AUC=0.921 (N=12680). All three kinases hit
≥ 0.87 on ECFP4 + XGBoost. BRAF was not exceptional — the cancer
kinase pipeline reproduces across targets, supporting H3-style
"pipeline generalizes" claim at the *predictive* level (interpretability
generalization still untested).

**L-20260529-17** — refers_to: P1 — polarity: qualifies — source: measured
*reports/20260529_braf_boosted result*. Adding null-pchembl rows as
inactives shifts BRAF from 40.9% active (5,751 mol) to 27.1% active
(8,671 mol). The boosted set is closer to the realistic "publication
bias mitigated" regime but treats unmeasured-pchembl rows as
definitively inactive — a strong assumption. Downstream pipeline runs
on the boosted set are not yet performed.

**L-20260529-31** — refers_to: P1 — polarity: qualifies — source: measured
*reports/20260529_braf_boosted_ecfp4.json* (boosted downstream — resolves
L-17's "not yet performed"). boosted-BRAF (8671 mol, 27.1% active) scaffold
OOF AUC **0.935** vs default-BRAF 0.909, AUPRC 0.85 vs 0.878, SAE R² 0.909.
Counterintuitively the metric IMPROVES — but this is the **easy-negative
effect**: adding 2920 null-pchembl molecules as inactives dilutes with
easy-to-classify negatives, inflating AUC. It does NOT mean better learning.
Lesson: publication-bias "mitigation" by null-as-inactive makes headline
AUC look better while the task becomes easier — report active% alongside
AUC always.

**L-20260529-32** — refers_to: H5 — polarity: qualifies — source: measured
*reports/20260529_{egfr,jak2}_time_sweep.json*. Cross-kinase time-split:
BRAF sharp 2014→2016 cliff (0.60→0.86), JAK2 gentle drift (0.53→0.75),
EGFR flat/near-random (~0.5, dips to 0.48). The temporal break is
BRAF-specific, not a universal kinase property. H5 generalizes only as
"run the per-target time-sweep" — the shape (cliff/drift/none) varies.

**L-20260529-33** — refers_to: H2 — polarity: qualifies — source: measured
*reports/20260529_sae_monosem_braf_ecfp4.json*. BRAF ECFP4 SAE: 2048
active latents, **0 monosemantic** (top selectivity 0.379). Descriptor
recovery R² (0.872, population metric) overstates per-feature
interpretability — no individual latent cleanly maps to one descriptor.
Resolves H2's open question: descriptor-R² ≠ monosemanticity. Uni-Mol
SAE monosemanticity (Bharadwaj's dense-embedding setting) remains the
open sub-question (needs GPU).

**L-20260529-18** — refers_to: H4 — polarity: supports (granularity-dependent) — source: measured
*reports/20260529_braf_mmp_mined.json*. Data-mined MMPs on BRAF with
delta-atom-count buckets (Tanimoto ≥ 0.85 + same scaffold): 8
transformations, 12.5% series-local. Same magnitude as default-SMIRKS
result (L-20260529-14). The granularity of the transformation key
determines whether series-locality is detectable — delta-atom-count
is too coarse, just like preset SMIRKS. MCS-level substituent keys
(mmpdb-grade) would test the Auer 2016 claim more directly.

**L-20260529-19** — refers_to: P1/P2 (kinase chemistry observation) — polarity: supports — source: measured
*Same BRAF MMP mining*. "Smaller is more potent" pattern: delta_-3
(B has 3 fewer heavy atoms than A) → 100% smaller-compound-active;
delta_+3 → 3.3% larger-compound-active. Consistent with BRAF ATP
pocket constraints. Suggests an evidence-driven heuristic for hit
optimization on this target.

**L-20260529-20** — refers_to: H2, H6 — polarity: refutes/qualifies — source: measured
*reports/20260529_braf_unimol_minimal.json*. RTX A6000 EU-SE-1, $0.19
actual cost. BRAF Uni-Mol scaffold-fold OOF AUC=0.804, SAE
(latent=4096, epochs=100) dead_ratio=0, R²_median=0.852, R²>0.5: 10/10.
Per-descriptor: MolWt 0.901, HeavyAtomCount 0.912, TPSA 0.882,
FractionCSP3 0.887, NumRotatableBonds 0.863, NumHAcceptors 0.840,
NumHDonors 0.814, MolLogP 0.646, NumAromaticRings 0.619,
RingCount 0.594. H2 falsified — Uni-Mol R² is essentially the same as
ECFP4 R² on the same SAR-narrow data. The "diversity drives SAE R²"
prediction does not hold for either feature space at these scales.

**L-20260529-21** — refers_to: H6 (new) — polarity: refutes — source: measured
*Same run*. Uni-Mol scaffold AUC 0.804 < ECFP4 scaffold AUC 0.909
(gap −0.105). Uni-Mol *underperforms* ECFP4 on BRAF. The implicit
"3D > 2D fingerprint" framing of the tutorial fails on target-
specific cancer-kinase data — likely because BRAF SAR is dominated by
2D pharmacophore patterns that ECFP4 captures directly while Uni-Mol's
general-chemistry pretraining averages target-specific discrimination
out. README's CO-ADD-based "Uni-Mol beats ECFP4" claim does not
generalize to single-target kinase.

**L-PENDING-Uni-Mol-CO-ADD** — refers_to: H1, README claim — polarity: pending — source: deferred
CO-ADD Uni-Mol reproducibility (the EXPECTED_OUTPUTS 0.895 AUC, 0.824
SAE R² values). Deferred to a future GPU session — this session's
budget cap was hit by BRAF Uni-Mol alone.

## Compaction policy

When entries exceed 30, the oldest VALIDATED entries (with subsequent
confirming evidence) can be summarized into a single rollup entry. The
LANDSCAPE registry is for traceable evidence, not a journal.

**L-20260529-22** — refers_to: H3 — polarity: qualifies — source: measured
*reports/20260529_{egfr,jak2}_external_ecfp4.json* (GH #3). Cross-target
BindingDB external eval: BRAF gap +0.109, EGFR gap +0.052, JAK2 gap
−0.060. H3 is target-dependent — the JAK2 reversal (external easier) is
driven by its 14.3%-active external set inflating AUC. Lesson: external
validation is mandatory *because the gap is unpredictable*, not because
it always drops. EGFR/JAK2 BindingDB fetched after server recovery
(prior session HTTP 500).

**L-20260529-23** — refers_to: H4 — polarity: supports — source: measured
*reports/20260529_braf_mmp_mined_mcs.json* (GH #4). MCS substituent-level
MMP keys (RDKit rdFMCS): 56 transformations, **58.9% series-local** (33/56)
vs 12.5% with atom-count-delta keys. H4 verdict flips FALSIFIED→VALIDATING.
The earlier null was a key-granularity artifact — coarse keys collapse
distinct substituent swaps into one bucket, washing out scaffold-locality.
Reproduces Auer 2016 at the granularity they used. Implemented as
`scripts/mmp_mine.py --key-mode mcs`.

### TYMS extension (H7 prep, 2026-05-29)

**L-20260529-28** — refers_to: P1, H7 — polarity: qualifies — source: measured
*scripts/diagnose_dataset.py on data/processed/tyms.csv*. Human TYMS
(CHEMBL1952): 874 records → 637 unique molecules. pChEMBL median 6.12
(vs BRAF 7.68) — a weaker-affinity enzyme literature. Active% by cutoff:
≥6.0=54.9%, ≥6.5=38.3%, ≥8.0=9.1%. **P1's kinase default (≥8.0) yields
only 9.1% active (~58 mol) — too sparse; rejected. Cutoff set to 6.0
(54.9%, balanced).** Confirms tcrit T-cutoff: P1 is kinase-specific, not
universal. 183 scaffolds, top-10 share 43.2% (more diverse than a single
kinase). Activity cliffs 5.8% — *below* H1's 0.10 threshold, between
CO-ADD (0.1%) and BRAF (11.1%): TYMS is a mid-regime enzyme, a useful
third datapoint for H7 discrimination.

**L-20260529-29** — refers_to: H7, H1 — polarity: supports — source: measured
*reports/20260529_split-compare_ecfp4.json (tyms)*. ECFP4+XGBoost on
TYMS: random-split AUC 0.878, **scaffold-split AUC 0.800 (fold_std
0.069)**, gap +0.079. The scaffold AUC 0.800 is the bar H7's finetuned-
Uni-Mol lane must beat. The fold_std 0.069 is large — empirically
confirms tcrit T-variance: a finetuning "win" must exceed ~0.07 to be a
real finding, not seed/fold noise. Random−scaffold gap +0.079 > H1's
0.05 → scaffold split is the honest metric here too.

**L-20260529-30** — refers_to: H7 (T-ft-infra) — polarity: supports — source: code-audit
*unimol_tools.MolTrain (train.py:37; data/split.py:45)*. MolTrain
natively supports `split='scaffold'` (Bemis-Murcko + GroupKFold,
identical discipline to the tutorial's `scaffold_folds`) and
`split='select'` (precomputed fold ids 0..k−1), plus `freeze_layers` for
freeze-depth control. The finetuning lane can therefore reuse the *exact
same* scaffold folds as the ECFP4/frozen lanes — scaffold parity
(tcrit's #1 critical blocker) is achievable, not merely asserted.

**L-20260529-31** — refers_to: H6, H7 — polarity: supports — source: measured
*reports/20260529_split-compare_unimol.json (tyms)*. Frozen Uni-Mol
embedding (512-d, CPU) + XGBoost on TYMS: random AUC 0.781, **scaffold
AUC 0.731 (fold_std 0.035)**. Versus ECFP4 scaffold 0.800 → **gap
−0.069: frozen Uni-Mol loses to ECFP4 on TYMS too.** H6 (frozen <
ECFP4) now reproduces on a *non-kinase, mid-regime enzyme*, not just
BRAF — the frozen-embedding deficit is general, not kinase-specific.
Crucially this means tcrit's Axis-4 contingency ("if frozen wins on
the enzyme, H7 has no loss to recover on TYMS") did NOT trigger: there
is a real −0.069 gap to close, so TYMS stays a valid H7 testbed. The
H7 test is now crisply framed: does finetuning lift Uni-Mol from 0.731
to ≥0.80 (matching ECFP4) on identical scaffold folds?

**L-20260529-24** — refers_to: H1, H6 — polarity: supports (weak) / refutes — source: measured
*reports/20260529_braf_unimol_splits.json* (GH #5). BRAF Uni-Mol:
stratified AUC 0.832, scaffold AUC 0.806, gap 0.026 (ECFP4 gap was
0.019). H1 weakly-validated on dense embeddings too — gap stays < 0.05,
and is NOT larger than ECFP4 as speculated. H6 reinforced: Uni-Mol
loses to ECFP4 on both splits (-0.096 stratified, -0.103 scaffold).
Two scaffold runs (A6000 0.804, 3090 0.806) agree to 0.002.

**L-20260529-25** — refers_to: README claim, H1 baseline — polarity: supports — source: measured
*reports/20260529_coadd_unimol_repro.json* (GH #2). CO-ADD Uni-Mol
reproducibility on fresh RTX 3090: OOF AUC 0.894 (baseline 0.895),
AUPRC 0.259 (0.256), SAE R²med 0.826 (0.824), dead 0.000. All within
regression tolerance (±0.01 AUC, ±0.05 R²). The EXPECTED_OUTPUTS
headline numbers are reproducible and GPU-class-independent.
