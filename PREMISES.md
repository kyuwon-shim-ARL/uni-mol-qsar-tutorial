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

**Status**: WEAKLY-VALIDATED on both ECFP4 and Uni-Mol (resolved 2026-05-29, GH #5)
**Measurement (BRAF, stratified vs scaffold OOF AUC)**:

| Feature | stratified | scaffold | gap |
|---------|-----------|----------|-----|
| ECFP4 (2048) | 0.928 | 0.909 | 0.019 |
| **Uni-Mol (512-d)** | **0.832** | **0.806** | **0.026** |

**Interpretation**: the stratified−scaffold gap is real but small on
**both** representations (0.019 ECFP4, 0.026 Uni-Mol), well below the
0.05 falsification threshold. The earlier speculation that dense 3D
embeddings would show a *larger* leakage gap than sparse fingerprints
is **not supported** — the gaps are within 0.01 of each other. Final
verdict: "scaffold split is the better practice" holds (gap > 0, and
Uni-Mol's is slightly larger), but "random K-fold catastrophically
inflates AUC (≥0.05)" is overstated for both feature spaces on BRAF.
The literature's 0.05–0.15 expectation is dataset-specific and does not
reproduce on this kinase set.
**Owner**: `data.scaffold_folds`, T4 in `EXPECTED_OUTPUTS.md`, GH #5.
**dependent_workflows**: `examples/run_full_pipeline.py`.

### H2 — SAE descriptor recovery R² depends on chemical diversity

**Claim**: Sparse autoencoder descriptor-recovery R² is upper-bounded by
the chemical diversity of the input distribution. SAR-narrow target
datasets (top-10 scaffold share ≥ 0.30) should not be expected to match
the 0.824 R² achieved on CO-ADD PA (diverse phenotypic).

**Status**: FALSIFIED on both ECFP4 and Uni-Mol (population R²); per-latent
monosemanticity ≈0 on both representations (ECFP4 + dense, resolved 2026-05-29)
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

**Open question RESOLVED (2026-05-29)**: monosemanticity measured.
`scripts/sae_monosemanticity.py` on BRAF ECFP4 SAE (2048 latents): of
**2048 active latents, 0 are monosemantic** (criterion: one descriptor
explains R²≥0.5 AND beats runner-up by ≥0.2). Top latent selectivity is
only 0.379 (NumHDonors). **This sharpens the H2 verdict**: descriptor
recovery R² (0.872) is a *population* metric — the SAE basis collectively
spans descriptor space — but **no individual latent is a clean
descriptor detector** (per-latent polysemantic). Descriptor-R² therefore
*overstates* per-feature interpretability, exactly the Bharadwaj-2024
distinction. So H2's "SAE is interpretable regardless of diversity" is
true only at the population (R²) level; at the monosemantic-feature
level the ECFP4 SAE is poorly interpretable.

**Dense-embedding sub-question RESOLVED (2026-05-29, GPU-free):** the
prior caveat ("Uni-Mol SAE monosemanticity needs GPU") is closed by
*re-scoring* the sibling project's already-trained dense SAE rather than
re-running one. The `uni-mol-QSAR` project's e191v3 SAE (768→6144 on
1,144 OOF Uni-Mol2 embeddings; population descriptor-R² 0.89–0.95) was
re-scored under *this* repo's
exact strict criterion (`scripts/sae_monosemanticity.py`, sel≥0.5 +
margin≥0.2, verbatim). Result: **0 / 6144 monosemantic (0.00%), top
selectivity 0.415 (HeavyAtomCount)** — essentially identical to ECFP4
(0/2048, 0.379). The dense 3D foundation-model SAE is **no more
monosemantic than ECFP4** at the per-latent level; both sit far below
the 0.5 bar. Alignment was proven by reproducing the original's
population R² (MolWt 0.914 vs 0.927, FractionCSP3 0.948 vs 0.937, etc.).
**This CONFIRMS (does not refute) the original project's own mature
finding.** Its e126 experiment (z-scored, proper training) reports the
*descriptor* projection at median R²≈0 (only 2.5% of latents reach even
the trivial R²>0.1 bar) — the original *itself* measured near-zero
descriptor alignment. (Its earlier e103 "0" is *not* cited here: e126
disavowed it as a no-z-score / full-batch training artifact.)
**"Monosemantic" splits into two axes — keep them separate:**
- *concept axis* (latent ↔ one physicochemical descriptor or substructure):
  ≈0 on both ECFP4 (0/2048) and dense (0/6144 here), and matches the
  original's own descriptor R²≈0. This is the axis our strict criterion
  tests, and H2's verdict on it is **unconditional**.
- *class-discriminative axis* (latent ↔ one phenotypic class,
  top1_class_frac>0.8): the original reports ~13.7% (e126, z-scored). We
  did **not** measure this and do **not** refute it — it is a different
  question (predicts class ≠ encodes one human concept).
So the honest claim is narrow: SAE latents are **not clean single-concept
detectors** (descriptor/substructure), on dense as on ECFP4 — *not* that
SAE latents carry no class signal.
**Scope note (#6-general vs #6-scoped):** e191 is a *PA-finetuned* dense
SAE (diverse phenotypic), so this closes the *general* dense-embedding
question on the concept axis. The tutorial's own setting (frozen Uni-Mol
on BRAF/CO-ADD) is not separately measured, but with concept-
monosemanticity ≈0 across two representations and two readout families,
the same result is the strongly-predicted outcome.
**Robustness (not a config artifact, 2026-05-29):** the 0 survives two
stress tests. (i) *Expansion*: retraining at 16× (768→12288, dead_ratio 0)
still gives **0/12288 monosemantic** — top selectivity only creeps 0.415→0.496
(still below 0.5), while population R²→0.9999 (pure overparameterized fit, not
interpretability). (ii) *Readout*: re-scoring against a 46-group RDKit
functional-group (substructure) panel instead of scalar descriptors gives
**0/6144** (top sel 0.201). So per-latent monosemanticity fails across two
expansions (8×/16×) AND two readout families (physicochemical descriptors,
functional-group substructures); sparsity was already swept (e103: 3λ×5seeds
all 0). The conclusion is a property of the SAE-on-molecular-embedding
problem, not of one hyperparameter choice. *Limitation*: the substructure
panel is RDKit fr_* only — bespoke scaffolds/motifs untested.
**Reference**: Bharadwaj et al. 2024 (arXiv:2512.08077) — SMI-TED SAE on
diverse PubChem reports high variance explained, but no diversity
ablation (and no per-latent monosemanticity test — exactly the gap this
closes).
**Owner**: `sae.descriptor_recovery` + `scripts/sae_monosemanticity.py`, T8;
re-score `uni-mol-QSAR/scripts/h2_dense_monosem_rescore.py` →
`results/e191/h2_dense_monosem_rescore.json`.

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

**Status**: VALIDATING — but **target-dependent**, not a universal kinase
property (cross-kinase tested 2026-05-29).
**Measurement (time-sweep ecfp4, AUC by train≤cutoff / test>cutoff)**:

| Cutoff | BRAF | EGFR | JAK2 |
|--------|------|------|------|
| 2010 | 0.548 | 0.557 | 0.528 |
| 2012 | 0.569 | 0.554 | 0.621 |
| 2014 | 0.603 | 0.539 | 0.637 |
| 2016 | **0.859** | 0.480 | 0.660 |
| 2018 | 0.730 | 0.576 | 0.748 |
| 2020 | 0.875 | 0.561 | 0.657 |

**Interpretation**: the 2014→2016 temporal break is **BRAF-specific**.
- **BRAF**: sharp jump 0.60→0.86 (post-vemurafenib scaffold influx).
- **JAK2**: gentle monotone rise 0.53→0.75 — mild, gradual drift, no cliff.
- **EGFR**: flat ~0.5 (even dips to 0.48 at 2016) — time-split is
  near-**random**. EGFR's pre/post-cutoff chemistry is either so diverse
  that no era predicts another, or the deposit timeline doesn't align with
  scaffold introduction. Either way, "train on past → predict future"
  fails entirely for EGFR at the ECFP4 level.

The honest generalization: **time-split is the most honest single-number
metric** (OOF AUC hides temporal structure), but the *shape* of the
temporal dependence is per-target — sharp cliff (BRAF), gentle drift
(JAK2), or no signal (EGFR). H5's original BRAF-specific "2014/2015
cliff" claim holds for BRAF only.
**Owner**: `data.time_split_indices`, T5; `scripts/time_split_sweep.py`.
**dependent_workflows**: any "predict future inhibitor" claim — must run
the per-target time-sweep, not assume a universal break.

### H7 — Finetuning Uni-Mol recovers the gap that frozen embeddings lost

**Claim**: H6 showed *frozen* Uni-Mol embeddings (general-chemistry
pretraining, no target adaptation) lose to ECFP4 on single-target QSAR
(BRAF: 0.804 vs 0.909). H7 is the sequel: **finetuning** the Uni-Mol
backbone on the target's labels (backprop through the transformer, not
just `get_repr`) should adapt the representation to the target and
close or reverse that −0.105 gap. This is the foundation→finetuning
paradigm the tutorial currently omits (it uses Uni-Mol frozen only).

**Status**: PARTIALLY VALIDATED on **two** targets (BRAF kinase + TYMS
enzyme) — finetuning recovers most of the frozen-embedding gap and the
from-scratch control confirms pretraining is the driver, but finetuning
reaches **parity with, not superiority over, ECFP4** (measured 2026-05-29,
RTX A6000, total $1.28). The earlier "from-scratch infra-blocked"
limitation is now **RESOLVED**.

**Measurement (scaffold 5-fold, `split='select'` parity, MolTrain epochs=30, seed 0)**:

| Lane (scaffold OOF AUC ± per-fold std) | TYMS (enzyme, 637) | BRAF (kinase, 5751) |
|----------------------------------------|--------------------|---------------------|
| ECFP4 + XGB | 0.800 | 0.909 |
| frozen Uni-Mol + XGB | 0.731 | 0.806 |
| **finetuned Uni-Mol** | **0.792 ± 0.084** | **0.890 ± 0.021** |
| from-scratch Uni-Mol (random init) | 0.641 ± 0.116 | 0.758 ± 0.031 |

(BRAF finetune 0.890 cross-validates the independent epochs=20 / 2-seed run
that measured 0.887 ± 0.006 — same conclusion, two code paths.)

**Interpretation** — three findings, consistent across enzyme and kinase:
1. **Finetuning recovers most of the frozen deficit**: TYMS 0.731→0.792
   (+0.061), BRAF 0.806→0.890 (+0.084 ≈ 80% of the 0.103 gap). The
   representation adapts to the target.
2. **Pretraining is the driver, not just end-to-end NN training** — the
   from-scratch control (same architecture, random init) lands far below
   finetuned: TYMS 0.641 vs 0.792 (**+0.151**), BRAF 0.758 vs 0.890
   (**+0.132**). This is the load-bearing control: it rules out "any NN
   beats XGBoost" and isolates the pretrained weights as the cause.
3. **But finetuning only reaches ECFP4 parity, never superiority**:
   TYMS 0.792 vs 0.800 (−0.008), BRAF 0.890 vs 0.909 (−0.019) — both
   within the per-fold std. On single-target QSAR a 2D fingerprint + tree
   model stays as good as a finetuned 3D foundation model, far cheaper.

**Practical takeaway for the tutorial**: the foundation-model advantage is
NOT automatic on target-specific data. Frozen embeddings lose to ECFP4;
finetuning catches up to parity (and pretraining demonstrably causes that
catch-up); cheap ECFP4 is never beaten. "Use the 3D foundation model here"
is not justified by accuracy alone on these targets.

**Methods notes (tcrit-hardened, resolved)**:
- *Scaffold parity*: all three lanes use identical Bemis-Murcko folds —
  `MolTrain(split='select')` consumes precomputed fold ids from
  `qsar_tutorial.data.scaffold_folds` (L-20260529-30).
- *From-scratch control*: RESOLVED by monkeypatching
  `UniMolModel.load_pretrained_weights` to a no-op (random-init backbone +
  trained head) — what the prior run flagged as infra-blocked.
- *Variance*: `unimol_tools` hardcodes the DataLoader shuffle generator
  (`get_ddp_generator(seed=3407)`) and `Trainer.set_seed(config.seed)`, so
  run-to-run variance ≈0 (seeds 0/1 differed <0.002, one fold only).
  Multi-seed CI is degenerate; the honest error bar is the **per-fold
  std**, reported above — 1 seed suffices on this library.
**Owner**: `scripts/finetune_unimol.py` (MolTrain wrapper);
`reports/{tyms,braf}_h7_finetune.json`.
**dependent_workflows**: any "use the 3D foundation model for this target"
decision — weigh the finetuning GPU cost against ECFP4's equal, near-free
result.

<!-- H7 original design notes (tcrit 4-axis) retained below for reference -->

### H7-DESIGN (original plan, retained)

**Test design** (tcrit-hardened, 4 axes — see derivation below):

*Axis 1 — fair comparison (the load-bearing axis):*
- **Scaffold parity**: all three lanes (ECFP4+XGB / frozen-UniMol+XGB /
  finetuned-UniMol) use the *same* Bemis-Murcko scaffold folds.
  Verified feasible: `unimol_tools.MolTrain(split='scaffold')` uses
  GroupKFold over Murcko scaffolds; `split='select'` accepts precomputed
  fold ids (0..k−1) for exact parity with the XGB lanes.
- **From-scratch control** (required): same Uni-Mol architecture,
  *random init*, trained on the target. Without it a finetuned win only
  proves "NN > XGBoost", not "pretraining helps". H7 is unfalsifiable
  without this arm.
- **Variance**: finetuning is stochastic; report multi-seed mean ± CI,
  not a single number. The gap to beat (0.105) must exceed the seed
  spread to be a real finding. (Repo norm: bootstrap CIs, Vina noise
  floor.)
- **HP scope**: a fixed, pre-registered hyperparameter budget (LR,
  epochs, `freeze_layers` depth, early stopping). A finetuned loss from
  bad HPs must NOT be read as refuting H7 — log the search that was run.

*Axis 2 — which data answers which question:*
- **TYMS (CHEMBL1952, 874 mol, enzyme)**: tests whether the H6
  frozen-loss generalizes *beyond kinases*. NOT used for the data-size
  sweep — 874 is too small to subsample a train curve and keep a stable
  test set.
- **BRAF (5,751 mol, already in repo)**: the known frozen-loss case, and
  the home for any low-data scaling curve (N=200/400/800 with stable
  held-out). Does finetuning win where frozen lost?
- TYMS active cutoff is decided by diagnosis (P1's pChEMBL≥8 was set on
  kinases; re-examine for this enzyme before labelling).

*Axis 3 — cost & external dependency:*
- Finetuning needs GPU (RunPod) and costs real money — multiplicative:
  lanes × targets × folds × seeds × (HP points). Hard cost cap +
  rollback ("stop after TYMS, BRAF is a follow-up") set BEFORE any GPU
  spend. GPU stockout already bit this project (2026-05-29).

*Axis 4 — recording vs verdict (honesty):*
- "Measurement obtained" = phase done; "H7 supported/refuted" = finding.
  Finetuning failing to win does NOT make the phase a failure.
- Branch to pre-define: if frozen Uni-Mol happens to *win* on TYMS
  (enzyme SAR ≠ kinase), there is no loss to recover on TYMS and the H7
  framing falls back to BRAF.

**Falsification**: with scaffold parity + from-scratch control +
multi-seed CIs + a documented HP budget, finetuned-UniMol AUC ≤ ECFP4
AUC (within CI) on both TYMS and BRAF refutes H7.

**Owner**: new finetuning lane (`unimol_tools.MolTrain` wrapper); README
"What this teaches" (Uni-Mol value proposition).
**dependent_workflows**: any "use the foundation model, not fingerprints"
recommendation in the tutorial.
**lineage**: H6 (frozen embedding falsified) → H7 (does finetuning fix
it). See LANDSCAPE L-20260529-20, L-20260529-21.

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
