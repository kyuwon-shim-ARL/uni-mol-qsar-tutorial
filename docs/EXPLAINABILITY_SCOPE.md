# Explainability scope (ligand-side)

This document defines what the tutorial's three interpretation layers
(SHAP / SAE / MMP counterfactual) *can* and *cannot* answer.

When extending the pipeline from CO-ADD PA (phenotypic, antibacterial) to
a cancer target like BRAF (target-specific, kinase binding), the same code
runs but the **meaning** of the output changes. Without setting scope, a
user can read protein-mechanism answers into ligand-only results.

## In scope — ligand-side questions the pipeline answers

1. **Which chemical features predict the label?**
   Answered by TreeSHAP on Uni-Mol features. Output: per-molecule and
   global attribution for each of the 512 Uni-Mol latent dimensions.

2. **What sparse, descriptor-like axes structure the embedding space?**
   Answered by the sparse autoencoder (SAE) + descriptor recovery R².
   Output: ~4096 SAE latents, of which a subset correlates linearly with
   interpretable RDKit descriptors (logP, TPSA, MW, donor count, etc.).

3. **What single-edit substructure changes shift the predicted activity?**
   Answered by MMP counterfactual: apply a transformation (e.g.
   `add_COOH_aromatic`) to each inactive parent, re-score, report Δp.

These are all **ligand-side**: they describe properties of the molecule,
not properties of the protein binding pocket.

## Out of scope — questions the pipeline does NOT answer

1. **Why does this molecule bind BRAF specifically?**
   Requires the protein structure (PDB). Use docking (AutoDock Vina,
   OpenFold + GNINA) or co-crystal analysis. Not in this tutorial.

2. **Which residue in the BRAF kinase pocket drives selectivity?**
   Same answer: structure-based methods.

3. **Will this active in our model survive a cellular assay?**
   QSAR predicts activity against the *training-set definition*. For
   BRAF that is "ChEMBL pchembl_value ≥ 8 in some biochemical assay" —
   not cellular efficacy, not selectivity, not ADME. The model knows
   nothing about these.

4. **Are the discovered MMP rules transferable to other kinases?**
   See *Limit 1* below — kinase MMP rules are typically scaffold-local.

## Limits documented in the literature

Three load-bearing limits, each with a citation. Read these before
reporting any interpretation as a "finding."

**Limit 1 — MMP rules from kinase data are scaffold-local.**
Auer et al. (2016, J. Chem. Inf. Model., PMC5198793): the distribution
of activity changes for a fixed transformation across scaffolds is
"nearly symmetrical and centred at zero" — a rule that is significant
in one scaffold is statistically uninformative in another. Best practice
(Kramer's method, see PMC12107391): group MMP pairs by Murcko scaffold
*before* computing rule statistics; report rules as "series-specific"
unless they replicate across multiple scaffolds. The tutorial's MMP
output (`counterfactual.py`) now reports `n_distinct_scaffolds` per
rule (see T7 ticket).

**Limit 2 — kinase data has unusually high activity-cliff density.**
PMC11032345 (2024) catalogued BRAF-specific activity cliff generators
(sixteen significant ones). Both XGBoost AUC and SHAP attributions
degrade on cliff compounds — van Tilborg et al. (2022, PMC9749029)
showed this is a property of the data, not the model. Practical
consequence: scaffold split is mandatory for kinase data
(`data.scaffold_folds`); random K-fold inflates AUC by 0.05–0.15.

**Limit 3 — SAE feature density depends on input chemical diversity.**
No published numbers on Uni-Mol SAEs at any diversity level (gap noted
in /lit search 2026-05-28). Closest reference: Bharadwaj et al. (2024,
arXiv:2512.08077) trained an SAE on IBM SMI-TED embeddings of 5M
PubChem molecules (very diverse) and reported 2,501 active features
out of 6,144 (8× expansion). On a SAR-narrow target dataset like BRAF
the count of *monosemantic* features is expected to be lower because
the input embedding distribution is narrower — fewer independent
directions to disentangle. Concrete prediction: descriptor recovery R²
on BRAF will be lower than the 0.824 measured on CO-ADD PA. Treat any
single SAE R² number as conditional on the input chemical space.

## Reporting rule

When writing up an interpretation result on target-specific data,
state the scope explicitly:

> "These SHAP / SAE / MMP outputs describe **ligand-side** structure–
> activity patterns within the BRAF ChEMBL set. They do not localize
> protein binding, do not establish kinase selectivity, and discovered
> MMP rules are scaffold-local unless `n_distinct_scaffolds ≥ K`
> (K=3 working threshold)."

The HTML report template (`report.py`) will be updated to embed this
scope line automatically when the run targets a single-protein dataset
(detected via the data manifest).
