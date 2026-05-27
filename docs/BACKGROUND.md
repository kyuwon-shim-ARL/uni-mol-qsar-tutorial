# Background — A Field Map for Interpretable Antibacterial QSAR

> **Audience**: research interns starting on molecular property prediction +
> model interpretation.
> **Purpose**: place this tutorial inside the larger field so you know which
> problem each tool was invented to solve, and where ours sits.
> **Reading order**: each section explains the *problem the previous generation
> could not solve*. If you only have 10 minutes, read §1, §4, and the table in §6.

---

## 1. The two questions and why they are hard

Drug discovery for small-molecule antibacterials boils down to two questions:

1. **Predict**: given a molecule (as a SMILES string), will it kill the
   bacterium?
2. **Explain**: if yes, *which part* of the molecule is doing the work, and
   *what change* would make it better?

Both are hard, but for different reasons.

| Question | Why it is hard |
|---|---|
| Predict | Activity depends on many things at once: target binding, getting *into* the cell, not getting pumped *out*, and not being toxic. None of these are reliably encoded in a 2D drawing. |
| Explain | Modern models are accurate but opaque. A 100M-parameter transformer that scores a molecule does not tell you "the carboxyl group is what matters". Interpretation is a separate research problem. |

For Gram-negative pathogens like *Pseudomonas aeruginosa* (**PA**) the predict
problem gets a third layer: the double membrane plus active efflux pumps. The
Lipinski Rule-of-5 — the classical drug-likeness filter — *does not apply* here.
That is the negative space in which this whole field operates.

---

## 2. How to read the generational map

Each generation of methods exists because the previous one could not solve
something. For each generation we list:

- **Pioneer** — the first paper that showed the idea worked at all.
- **Champion** — the version that everyone now benchmarks against (often years
  later than the pioneer, with better data or scale).
- **What it fixed** — the previous-generation failure mode it addressed.
- **What it could not solve** — the failure mode that motivated the next
  generation.

The pioneer is usually cited for credit; the champion is what you actually run.
They are rarely the same paper.

---

## 3. Generational map — five generations

### Gen 1 (1960s–1990s): hand-crafted descriptors + linear models

| Slot | Reference |
|---|---|
| Pioneer | Hansch & Fujita, 1964 — multiple linear regression on log P, σ, Es |
| Champion | Lipinski Rule of 5, 1997 — four hand-picked descriptors as a pass/fail filter |
| What it fixed | Replaced pure intuition with a number you could compute |
| What it could not solve | Nonlinear structure–activity relationships; novel scaffolds; antibacterial Gram-negative (Ro5 famously fails here) |

**Plain English**: count a few things about the molecule (size, greasiness,
hydrogen bonds), put them in a regression. Works for property *trends*, breaks
on real chemical diversity.

### Gen 2 (1990s–early 2010s): molecular fingerprints + classical ML

| Slot | Reference |
|---|---|
| Pioneers | Morgan, 1965 (algorithm); Rogers & Hahn, 2010 (ECFP4 as we know it) |
| Champion | **ECFP4 + Random Forest** (Svetnik 2003) — still the universal baseline you must beat |
| Friends | SVM (Burbidge 2001), gradient boosting (Sheridan 2016), MACCS keys |
| Data | ChEMBL public release (2009); PubChem (2004) |
| What it fixed | Captured substructure presence/absence at scale; handled nonlinear SAR |
| What it could not solve | 3D conformation effects; *interpretation* (bit #1234 = "?"); poor generalization to genuinely new scaffolds |

**Plain English**: hash every 2–4-bond fragment into a bit-vector, throw it at a
forest. Fast, robust, hard to explain individual predictions in chemical terms.

### Gen 3 (2015–2019): graph neural networks (GNNs)

| Slot | Reference |
|---|---|
| Pioneers | Duvenaud et al. 2015 (Neural Fingerprints); Gilmer et al. 2017 (Message Passing NN) |
| Champion | **Chemprop / D-MPNN** (Yang et al. 2019); used by Stokes et al. 2020 to discover halicin |
| What it fixed | Learned features end-to-end from the molecular graph instead of hand-designed bits |
| What it could not solve | Still 2D; gradient-based attribution on graphs is noisy; data efficiency on small antibacterial sets is limited |

**Plain English**: treat the molecule as a graph, let the network learn its own
"fingerprint." A genuine win for prediction. Interpretation is harder, not
easier.

### Gen 4 (2020–2023): 3D self-supervised foundation models

| Slot | Reference |
|---|---|
| Pre-pioneers | SchNet (Schütt 2017); 3D-Infomax (Stärk 2022); GraphMVP |
| Champion | **Uni-Mol** (Zhou et al. 2023) → Uni-Mol2 (2024) — 84M–1.1B parameter 3D transformer pretrained on 200M conformers |
| Friends | MolFormer (IBM 2022) for SMILES; SMI-TED (IBM 2024) |
| What it fixed | 3D conformation awareness; transfer learning from cheap unlabeled conformers |
| What it could not solve | Still a black box; benchmarks contaminated by data leakage (Wallach 2018, MoleculeACE 2022 documented this); no built-in interpretation |

**Plain English**: pre-train a big 3D transformer on millions of molecules, then
fine-tune on your tiny labeled set. This is the *representation* layer we use in
this tutorial — Uni-Mol's 512-dim (v1) / 768-dim (v2) embedding per molecule.

### Gen 5 (2023–present): the interpretability layer

This generation is itself split into three sub-threads because no single method
won.

**Sub-thread A — post-hoc attribution**

| Slot | Reference |
|---|---|
| Pioneer | SHAP (Lundberg & Lee 2017) |
| Champion | **TreeSHAP on a tree-model surrogate** (Sheridan 2019; Lundberg et al. 2020) — exact, fast, the standard recipe is "deep features → tree model → TreeSHAP" |
| Friends | ALE (Apley & Zhu 2020); occlusion / perturbation |
| What it fixed | Per-prediction feature attributions with mathematical guarantees |
| Limitation | Operates on whatever features you give it. If features are not chemically meaningful, attributions are not either. |

**Sub-thread B — mechanistic interpretability via Sparse Autoencoders (SAE)**

| Slot | Reference |
|---|---|
| Pioneer (general) | Bricken et al. 2023 (Anthropic) — SAE on LLM activations |
| Champion (chemistry) | **Cohen 2025 (SMI-TED)** — first SAE applied to a molecular foundation model |
| Friends | Templeton et al. 2024 (Claude features); many follow-ups in 2024–2025 |
| What it fixed | Decomposes dense entangled embeddings into sparse, often-monosemantic features (one feature = one concept) |
| Limitation | Needs careful training (dead neurons, normalization); chemical interpretation of features is still an open art |

This tutorial is *the second* SAE-on-molecular-foundation-model setup in the
public literature (after Cohen 2025). The framing is "feasibility +
hypothesis-generating," not "validated rule."

**Sub-thread C — counterfactual / matched molecular pairs (MMPs)**

| Slot | Reference |
|---|---|
| Pioneers | Matched Molecular Pair Analysis — Hussain & Rea 2010; Griffen 2011 |
| Champion (antibacterial) | **Gurvic 2024** — MMPA on the CO-ADD Gram-negative set (~73K compounds) |
| What it fixed | Connects "this substitution increases activity by X%" to a *transformation* you can synthesize |
| Limitation | MMPs need enough paired data; rules are often series-bound, not universal |

---

## 4. Antibacterial specifics — why Gram-negative is its own field

Antibacterial QSAR for Gram-negative pathogens (*E. coli*, *PA*, *K. pneumoniae*,
*A. baumannii*) is a sub-field with its own pioneers and champions because
generic drug-likeness rules fail.

| Problem | Champion paper | What it gave the field |
|---|---|---|
| What gets *into* a Gram-negative cell | **Geddes et al. 2023 (*Nature*)** | Rules-based PA accumulator: formal charge + PSA + HBD threshold on n=345 |
| What gets *pumped back out* (MexAB-OprM efflux) | **Mehla et al. 2021 (mBio)** | 174-descriptor framework (QSAR + QM + MD docking) on Rempex peptidomimetics |
| MexB substrate prediction | **Mansbach 2020/2023** | Machine learning on efflux substrate sets |
| Public Gram-negative phenotypic data | **CO-ADD** (Community for Open Antimicrobial Drug Discovery, since 2015) | ~300K phenotypic screens, freely licensed for research |

**Why this tutorial uses CO-ADD**: it is the largest *public* PA phenotypic
dataset, it is well curated (Blaskovich et al. 2015–), and it has a clean
active/inactive label per compound. It is also the standard external-validation
set used by Gurvic 2024 and the foundation of modern Gram-negative MMPA work.

**Data access note (added 2026-05-27)**: the dataset this tutorial actually
consumes is the CO-ADD PA screening data **as mirrored in ChEMBL**
(`src_id=40`, `src_short_name=COADD`). CO-ADD releases its screens on ChEMBL
after a 24-month embargo, and the ChEMBL public REST API needs no
registration. `scripts/download_chembl_coadd.py` pulls five PA assays
(four active, one empty), classifies inhibition / MIC subsets, and
aggregates per molecule into the ML-ready CSV. The lineage and the active
call rule are documented in `data/README.md` and reproduced byte-for-byte
against the parent project (24,120 molecules, 689 active = 2.86%).

---

## 5. Our position — what we premise and what we claim

Read this part with the rest of the field in mind. We are *not* introducing a
new model class. We are picking from the menu above and testing whether one
specific stack tells us something useful.

### 5.1 Three premises (what we assume to be true going in)

| ID | Premise | Why it matters | What would break it |
|---|---|---|---|
| **P1** | Labels in our training set are trustworthy after Cleanlab correction. | If labels are noisy, every downstream number is suspect. | External validation disagreement rate >20% |
| **P2** | Cross-validation embeddings are leakage-free (fold-specific, never seen across folds). | This is the difference between AUC=0.95 (leaky) and AUC=0.83 (honest). | New data added without recomputing OOF |
| **P3** | A single frozen "foundation pipeline" is the baseline that all experiments compare against. | Apples-to-apples comparison only works if the substrate is fixed. | New foundation declared |

The leakage point (P2) is not academic. In our parent project, the same model
scored AUC=0.95 with leakage and AUC=0.83 without. *Always* check whether
embeddings used in cross-validation were computed inside or outside the fold.

### 5.2 Three hypotheses (what we tried to show)

| ID | Hypothesis (plain English) | Status | Honest caveat |
|---|---|---|---|
| **H1** | A 3D foundation model representation is *compatible with* atom-level interpretation — not necessarily *better at ranking* than the old ECFP4 + Random Forest baseline. | Passed for interpretation; ECFP4+RF still wins on enrichment ranking in our data. | We do **not** claim representation superiority. |
| **H2** | An SAE on top of the foundation embedding produces sparse features that recover physicochemistry, and yields at least one *hypothesis-generating* design rule (a substitution suggestion) via matched-pair analysis. | Passed at hypothesis-generating tier (median descriptor R²≈0.9, propargylamine rule Δp(active)=+0.077 [BCa 95% CI 0.029–0.131]). | Not a "validated" rule. Needs wet-lab to promote. |
| **H3** | Different interpretation methods (SHAP, ALE, SAE, occlusion) converge on the same features. | **Falsified.** They diverge (e.g. SHAP↔ALE Spearman ρ≈0.05, n.s.). Lesson kept; deprecated as a positive claim. | Divergence is the result; pretending convergence would be the mistake. |

This is the honest framing. The deprecation of H3 is a feature, not a bug —
publishing a negative result about cross-method convergence is itself useful for
the field.

### 5.3 Where we sit on the evidence pyramid

A useful mental model is a 4-tier evidence pyramid for chemistry claims:

| Tier | What it means | Our status |
|---|---|---|
| **L1** | In silico predictive (model says X is active) | **Done** — macro-AUC ≈ 0.83, external EF@1% > 5× |
| **L2** | In silico mechanistic candidate (we have a story for *why*, supported by chemistry/structure) | **Reaching** — propargylamine rule + SAE descriptor recovery |
| **L3** | Wet-lab validation (real assay confirms the prediction) | **Out of scope** for this tutorial (parallel track) |
| **L4** | In vivo / clinical | Not applicable |

**Important**: this tutorial teaches you to produce **L1 + L2** evidence
*responsibly*. Anything claimed beyond L2 in the parent project requires
wet-lab data we do not have, and we do not pretend otherwise.

### 5.4 Known limitations of this tutorial implementation

These are deliberately surfaced. Fixing any of them is a sensible first PR
for an incoming intern; we keep the warts visible rather than hidden.

| # | Limitation | Why it matters | Suggested fix |
|---|---|---|---|
| **L1** | **Random K-fold split**, not scaffold split. `StratifiedKFold` shares Bemis-Murcko scaffolds across folds. | CO-ADD contains analogue families; an analogue of a training molecule sitting in the validation fold inflates AUC. Parent project saw Δ≈0.13 between leaky and honest splits. The headline OOF AUC≈0.83–0.90 should be read as an optimistic upper bound. | Add `scaffold_folds()` next to `stratified_folds()` in `src/qsar_tutorial/data.py`; report both metrics. (Scaffold split work is tracked separately by the maintainer.) |
| **L2** | **No external validation set.** Premise **P1** ("Cleanlab labels are trustworthy") declares "external validation disagreement rate >20%" as its breaking condition, but no external dataset is wired in. P1 is therefore unfalsifiable inside this tutorial as it stands. | Drug-discovery claims that never face an external dataset are field-known to overstate. | Wire in one external set (Stokes 2020 *halicin*, Wong 2024 *abaucin*, or a ChEMBL non-CO-ADD PA extract). Report EF@1% on that set alongside OOF AUC. |
| **L3** | **Per-molecule active rule ignores censored MIC.** `coadd_pa_combined_per_molecule.csv` declares `active=1` if `min_mic_nM ≤ 40_000`, without re-checking `standard_relation`. Rows originally tagged `MIC > 32 µg/mL` can still satisfy the aggregated condition. | Inflates the active class. This is preserved deliberately to byte-match the parent project's CSV (24,120 / 689 active). A censored-aware rule would reduce actives to roughly 100. | Add a `--strict-mic` flag in `scripts/download_chembl_coadd.py` that excludes `standard_relation == ">"` rows before aggregation, and report both label sets. |
| **L4** | **Final SHAP / SAE interpretations come from a single full-data refit.** Fold-wise stability (Jaccard or rank correlation of top-K features) is not reported. | Calling a top SHAP feature a "discovery" without fold stability is statistically weak; the parent paper requires this kind of check. | Re-run TreeSHAP / SAE per fold; report top-K Jaccard across folds. Drop features below a stability threshold. |
| **L5** | **ECFP4 vs Uni-Mol comparison is not apples-to-apples.** The qualitative caveat in **H1** ("ECFP4+RF still wins on ranking") is supported only by a `--max-n 3000` smoke run; the full-N (24,120) ECFP4 baseline has never been recorded. | A canonical statement should rest on equal data, equal folds. | Run the full-N ECFP4 path once (CPU-only, ~1 h) and record the result. |
| **L6** | **Raw → 24,120 derivation script** lived only in the parent project until 2026-05-27. Recovered as `scripts/download_chembl_coadd.py` (this commit). The four lineage CSVs and the active-call rule are now reproducible inside this repo. | Tutorials whose data appears by magic are not tutorials. | Already addressed — `python scripts/prepare_data.py` is now end-to-end. Keep `data/README.md` in sync if assays change. |

These six items, fixed in order, would turn the tutorial from
"reproducible-with-asterisks" to fully honest. None of them invalidate the
*pedagogical* purpose, but each one is a real methodological loose end an
intern should learn to see.

---

## 6. MECE summary table — the four axes you always reason on

When reading any QSAR / interpretability paper, place it on these four
axes. The axes are roughly mutually exclusive and collectively exhaustive
(MECE): every method choice falls under one of these.

| Axis | Choices (oldest → newest) | What this tutorial uses |
|---|---|---|
| **1. Representation** | Hand-crafted descriptors → Morgan/ECFP fingerprints → GNN learned features → 3D foundation model embedding → SAE-decomposed sparse features | Uni-Mol(v1/v2) embedding, optionally SAE-decomposed |
| **2. Predictor** | Linear regression → SVM/RF → gradient boosting → end-to-end neural net → frozen-features + tree model | XGBoost on frozen Uni-Mol embedding (TreeSHAP-compatible) |
| **3. Interpretation** | Regression coefficients → feature importance → SHAP / ALE → SAE features → counterfactual / MMP | TreeSHAP + SAE + MMP counterfactual (three layers, deliberately) |
| **4. Data** | Curated literature → ChEMBL → MoleculeNet benchmarks → phenotypic screens (CO-ADD) → wet-lab cohort | CO-ADD PA binary (active/inactive) — public, licensed for research |

**Why three interpretation layers, not one**: because H3 (above) showed they
*disagree*. Reporting all three and showing how they diverge is more honest
than picking one and claiming consensus.

---

## 7. What this tutorial is — and is not

**This tutorial is:**

- A working, public-data version of one specific stack on the Gen-4/Gen-5
  boundary: Uni-Mol embedding → XGBoost → SHAP + SAE + MMP counterfactual →
  HTML report.
- An exercise in *honest evidence framing*: L1 and L2 only, with explicit
  caveats and a deprecated hypothesis kept in view.

**This tutorial is not:**

- A claim that this stack beats simpler baselines on every task. (It usually
  does not beat ECFP4+RF on pure ranking.)
- A drug discovery pipeline. The output is a hypothesis to test, not a
  candidate to file an IND on.
- A general-purpose framework. Every choice (CO-ADD, Uni-Mol size, XGBoost,
  6144-neuron SAE) was made for educational clarity, not optimality.

---

## 8. Glossary (one line each)

- **SMILES** — string notation for a molecule (e.g. `CCO` for ethanol).
- **ECFP4** — Extended Connectivity Fingerprint, radius 2 (i.e. 4 bonds
  diameter). The canonical bit-vector representation.
- **SHAP / TreeSHAP** — game-theoretic feature attribution; TreeSHAP is the
  exact, fast version for tree models.
- **ALE** — Accumulated Local Effects; an alternative to partial dependence
  plots, robust to feature correlation.
- **SAE** — Sparse Autoencoder; an overcomplete autoencoder with an L1 (or
  TopK) sparsity penalty, used to find disentangled "features".
- **MMP / MMPA** — Matched Molecular Pair (Analysis); pairs of molecules
  differing by a single substitution, used to estimate transformation effects.
- **CO-ADD** — Community for Open Antimicrobial Drug Discovery; the canonical
  public Gram-negative phenotypic dataset.
- **OOF** — Out-Of-Fold; predictions for held-out folds, used to avoid
  leakage between feature extraction and model evaluation.
- **EF@k%** — Enrichment Factor at k% — how many more actives you find in the
  top k% of the ranked screen than you would by chance. Standard virtual
  screening metric.

---

## 9. Pointers for further reading

- **Honest benchmarking**: Wallach & Heifets 2018 (J Cheminform); van Tilborg
  et al. 2022 (MoleculeACE).
- **Foundation-model representation**: Zhou et al. 2023 (Uni-Mol);
  Ross et al. 2022 (MolFormer).
- **Chemistry SAE**: Cohen 2025 (SMI-TED); Bricken et al. 2023 (Anthropic, for
  the original SAE-on-foundation-model recipe).
- **PA / Gram-negative specific**: Geddes et al. 2023 (*Nature*); Mehla et al.
  2021 (mBio); Gurvic 2024 (CO-ADD MMPA).
- **Why explanation is hard**: Rudin 2019 ("Stop Explaining Black Box Models");
  Lipton 2018 ("The Mythos of Model Interpretability").
