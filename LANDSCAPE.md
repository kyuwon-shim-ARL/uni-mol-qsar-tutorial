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

**L-PENDING-Uni-Mol** — refers_to: H2 — polarity: pending — source: blocked
Uni-Mol SAE descriptor R² on BRAF + CO-ADD reproducibility on Uni-Mol.
Blocked by RunPod stock shortage (see
`.omc/pods/20260529_qsar-braf-a40_stockout.yaml`). Retry when stock
recovers.

## Compaction policy

When entries exceed 30, the oldest VALIDATED entries (with subsequent
confirming evidence) can be summarized into a single rollup entry. The
LANDSCAPE registry is for traceable evidence, not a journal.
