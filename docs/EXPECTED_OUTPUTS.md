# Expected outputs — verified on RunPod A40 (2026-05-26)

This document records the actual numbers the tutorial produced on a real
GPU run, so future readers can tell if their setup is working or broken.

Anything here was measured, not estimated. Replication recipe is at the
bottom.

## Setup

| Item | Value |
|---|---|
| Pod | NVIDIA A40 48GB, SECURE, CA-MTL-1 |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Wall time | 27 min (most of it preflight + Uni-Mol weight download) |
| Cost | **$0.20** |
| Code | `kyuwon-shim-ARL/uni-mol-qsar-tutorial` @ first commit |
| Data | CO-ADD PA combined per-molecule CSV, N=24,120 unique mols, 2.9% active |

## Stage 4 — saturation (n=10, 100 SAE steps)

Tiny job to prove the GPU path works at all.

| Metric | Value |
|---|---|
| `torch.cuda.is_available()` | `True` |
| Device | NVIDIA A40 |
| Uni-Mol GPU featurize, n=10 | 4.59 s (459 ms/mol, includes model load) |
| SAE 100 steps, batch 2560 × 512d → 4096 latent | 0.52 s (5.2 ms/step) |
| Loss trajectory | 0.5925 → 0.0002 |
| GPU memory peak | 447 MB |

**PASS**: GPU engaged, loss converges, memory in budget.

## Stage 5 — cache-warm (3 × N=500)

Three repeats to confirm timing stabilizes after first-run warmup.

| Rep | featurize | XGB OOF | SAE 30ep | AUC | dead | GPU peak |
|---|---|---|---|---|---|---|
| 1 | 11.35 s | 2.23 s | 0.67 s | 0.733 | 0.000 | 1067 MB |
| 2 | 8.11 s | 2.23 s | 0.21 s | 0.706 | 0.000 | 777 MB |
| 3 | 7.86 s | 2.25 s | 0.22 s | 0.556 | 0.000 | 1308 MB |

- Rep 2 vs Rep 3: featurize 3 % delta, SAE 5 %, XGB 1 % — **PASS** (under 15 % threshold).
- AUC varies 0.56–0.73 because each rep samples a *different* random 500-mol
  subset of a 2.9 %-active dataset; variance is sampling noise, not a
  pipeline defect. Stage 6 with full N confirms.

## Stage 6 — calibrated long run (full N=24,120)

The number to actually believe.

### Predictive performance (5-fold OOF)

| Metric | Value |
|---|---|
| **AUC** | **0.895** |
| AUPRC | 0.256 |
| MCC @ 0.5 | 0.268 |

For comparison, on the same machine the ECFP4 baseline (N=3,000 subsample,
CPU-only smoke run) hit AUC=0.847. So Uni-Mol on full N beats ECFP4 here.
Do not over-read this — see `BACKGROUND.md` §5.2 (H1) on why a
representation winning at one slice does not generalize to "Uni-Mol is
better."

### SAE health (latent=4096, epochs=100, on full N=24,120 × 512-d Uni-Mol)

| Metric | Value | Health threshold | Status |
|---|---|---|---|
| dead_ratio | **0.000** | < 0.05 | ✅ |
| RDKit descriptor R² median | **0.824** | > 0.50 | ✅ |
| descriptors with R² > 0.5 | 7 / 10 | — | strong |

For reference, the parent project's full-scale SAE (e191v3, n=1,144,
4-class) reported R²_median = 0.897 — the same order of magnitude on a
larger, different-domain dataset.

### MMP design-rule scan (500 inactive parents, 8 transforms)

| Transformation | n applied | Δp(active) mean | 95% CI | Exemplar Δp |
|---|---|---|---|---|
| `add_COOH_aromatic` | 437 | **+0.165** | [+0.150, +0.180] | +0.775 |
| `add_OH_aromatic` | 437 | +0.085 | [+0.070, +0.099] | +0.621 |
| `add_Cl_aromatic` | 437 | +0.063 | [+0.047, +0.077] | +0.572 |
| `add_F_aromatic` | 437 | +0.046 | [+0.032, +0.060] | +0.572 |
| `add_NH2_aromatic` | 437 | +0.041 | [+0.028, +0.054] | +0.593 |
| `add_methyl_aromatic` | 437 | +0.030 | [+0.017, +0.043] | +0.556 |
| `amidine_to_propargylamine` | 27 | −0.036 | [−0.080, **+0.024**] | +0.637 |
| `primary_amine_to_propargyl` | 26 | −0.025 | [−0.090, **+0.035**] | +0.272 |

Read the **CI** before the mean. Six of the eight rules have 95% CIs that
exclude zero — they are *hypothesis-generating* (per `BACKGROUND.md`
§5.3, L2 only — not validated rules). The two propargylamine rules have
CIs that include zero on CO-ADD binary; the propargylamine rule from the
parent project was discovered on a 4-class W-subset (different label
space) and does not transfer cleanly to binary active/inactive — exactly
the cross-dataset distinction the tutorial is designed to teach.

## GPU utilization

During the SAE training phase, `nvidia-smi` sampling over 6 × 2 s showed:

- VRAM held steady: **5,423 MiB / 46,068 MiB** (11.8 %)
- GPU compute util: **19–25 % sustained**

The `gpu_sample_burst` tool labels this `CONSISTENTLY_IDLE` because its
threshold is roughly >70 %, but 22 % sustained on small-batch SAE is
expected — the kernel cost per batch is too small to saturate an A40.
Use this as a calibration point: if you see <5 % util on the same job,
the GPU is genuinely not being used and something is wrong (e.g.
`use_gpu=False` or CUDA-toolkit mismatch).

## Replication recipe

```bash
# On any PyTorch 2.x / CUDA 12 box with an NVIDIA GPU:
apt-get install -y libxrender1 libxext6 libsm6 libglib2.0-0
pip install unimol-tools xgboost shap rdkit jinja2 huggingface_hub
git clone https://github.com/kyuwon-shim-ARL/uni-mol-qsar-tutorial.git
cd uni-mol-qsar-tutorial
# Put CO-ADD PA CSV at data/raw/ (see data/README.md)
python scripts/prepare_data.py
PYTHONPATH=src python examples/run_full_pipeline.py \
    --features unimol --sae-latent 4096 --sae-epochs 100 \
    --out reports/coadd_pa_unimol_FULL.html
```

Expected wall time on a single A40: ~5–8 min for the actual pipeline
after Uni-Mol weights are cached.

## Common gotchas surfaced by this preflight

1. **`huggingface_hub` is not in `unimol-tools` dependencies.** Without
   it, weight download silently fails with `ImportError`. Already pinned
   in `requirements.txt`.
2. **`libXrender.so.1` missing on bare PyTorch images.** RDKit's
   `Chem.Draw` import chain breaks. `apt-get install libxrender1` fixes
   it.
3. **RunPod `create_pod_auto` with `cloudType=ALL`** returns 400 across
   all DCs as of 2026-05-26; the REST API now requires `SECURE` or
   `COMMUNITY` explicitly. Direct `create_pod` works.

For full pod incident log, see `.omc/pods/2026-05-26_qsar-tut-a40.yaml`.

---

## Artifact naming convention (T0)

Every report or per-experiment artifact written to `reports/` MUST follow:

```
{YYYYMMDD}_{experiment-tag}_{featurizer}.{ext}
```

- `YYYYMMDD`: UTC date the artifact was produced
- `experiment-tag`: short kebab name (e.g. `split-compare`, `cleanlab-audit`, `full-pipeline`)
- `featurizer`: `ecfp4` or `unimol`
- `ext`: `html` for human-facing report, `json` for machine-readable counts

Existing baseline artifacts (`coadd_pa_unimol_FULL.html` etc.) are kept as
historical references — they are cited in this document. New runs MUST NOT
overwrite them. If a numerical comparison needs the baseline to be
re-generated, output to a new dated name and append a "vs baseline" row.

Cycle 1 deliverables (T1/T4/T2):

| Artifact | Produced by |
|---|---|
| `{date}_split-compare_ecfp4.html` + `.json` | `examples/run_split_comparison.py --features ecfp4` |
| `{date}_split-compare_unimol.html` + `.json` | same, `--features unimol` (GPU; runpod gate) |
| `{date}_cleanlab_flagged.csv` | `examples/cleanlab_audit.py` (maintainer review needed) |
| `{date}_cleanlab_audit.json` | same |

## GPU preflight checklist (T0, runpod-mcp)

Before any GPU-touching launch (T1 unimol path, full pipeline, T4 unimol
stability) you must clear each of the following. Failure of any item
aborts the launch — saves money.

1. **Weight cache check** — `python scripts/check_unimol_cache.py` returns
   exit 0. If 1 or 2, fix locally before paying for a pod.
2. **`mcp__runpod-mcp__plan_gpu_job`** — pass `randomAccessTrainingGb` (data
   size after rdkit canonicalization). Use the returned
   `containerDiskInGb` recommendation verbatim. NV-only paths are slower
   for random-access training (~18× per e049 measurement).
3. **`mcp__runpod-mcp__create_pod_auto`** — single-GPU first, lowest cost
   class. `costSafetyConfirmed: true` ONLY after user explicitly confirms
   the cost line item.
4. **`mcp__runpod-mcp__run_preflight`** — must pass with `trainingSmokeCmd`
   for any NEW script (e.g. `python examples/run_split_comparison.py
   --features unimol --csv data/processed/coadd_pa.csv` — make sure it
   completes the featurize step on 100 mols and the OOF step on 1 fold
   before going full).
5. **`mcp__runpod-mcp__gpu_sample_burst`** at T+120s — must NOT be
   `CONSISTENTLY_IDLE`. If it is, abort (CPU fallback / NV-streaming /
   NotImplementedError).
6. **Termination** — `mcp__runpod-mcp__delete_pod` immediately after
   artifact rsync. NEVER stop (stop is still billed). NEVER spot.

The `.omc/pods/{date}_*.yaml` incident log MUST be appended after each
launch with at minimum: pod id, GPU class, wall-clock, $ cost, exit code.

## Cleanlab honest framing (T2)

The cleanlab A/B is set up to surface a possibly-negative result. The
**risk** is circular flagging: at 2.86% active rate (689/24,120), confident
learning can mark hard-to-predict actives as "noisy" simply because the
model predicts them poorly. Removing them then *appears* to help AUPRC.

Defenses encoded in `examples/cleanlab_audit.py`:
- Bootstrap CI95 on `Δ AUPRC = AUPRC(cleaned) − AUPRC(original)` (1000
  resamples by default) — wide CI signals "no real effect".
- Paired permutation p-value (1000 perms) — checks how often a random
  swap of A/B labels produces an equal or larger Δ in magnitude.
- Maintainer-only review of flagged actives (intern does NOT mark
  `looks-spurious` without sign-off; comment column is for observation
  only on the intern path).

If Δ CI95 brackets zero OR permutation p > 0.05, the conclusion is
"cleanlab did NOT improve labels at significance" — report exactly that.
Do not iterate flagging strategies to chase a positive result.

---

## Data regime — phenotypic vs target-specific

The numbers above were measured on the CO-ADD PA phenotypic screen. When
the pipeline is run on a target-specific dataset (e.g. BRAF / CHEMBL5145),
expected ranges shift. Do not compare BRAF results to CO-ADD numbers
directly — the data regimes differ on three axes.

### Dataset diagnostic — measured (2026-05-28)

Run `scripts/diagnose_dataset.py <csv>` for any new dataset before
trusting downstream metrics.

| Property | CO-ADD PA | BRAF (ChEMBL5145, pchembl ≥ 8) |
|---|---|---|
| N molecules | 24,120 | 5,751 |
| Active fraction | 2.9% | 40.9% |
| Unique Bemis-Murcko scaffolds | 10,826 | 2,004 |
| Top-1 scaffold share | 8.3% | 1.0% |
| Top-10 scaffold share | 16.5% | 8.5% |
| Activity-cliff fraction (Tanimoto≥0.85, opp. label) | 0.1% | **11.1%** |

The cliff fraction is the load-bearing signal here: it is the data-side
proxy for "how much SAR structure does this dataset carry?" Phenotypic
screens are essentially cliff-free; kinase data is cliff-rich.

### Expected metric differences

| Metric | CO-ADD measured | BRAF expectation | Why |
|---|---|---|---|
| Random-fold AUC | 0.895 | 0.85–0.95 | inflated by SAR leakage |
| **Scaffold-fold AUC** | not measured | **0.70–0.85** | the number to actually believe (see `data.scaffold_folds` docstring: 0.05–0.15 drop is normal) |
| AUPRC | 0.256 | 0.50–0.85 | naturally higher when active fraction is 41% |
| SAE descriptor R² (median) | 0.824 | **0.40–0.60 expected** | narrower chemical space → fewer monosemantic SAE directions (gap in literature on Uni-Mol SAE; lower-bound estimate from `EXPLAINABILITY_SCOPE.md` Limit 3) |
| MMP rules with 95% CI excl. zero | 6 / 8 | 3–6 / 8 expected | cliff-driven rules look statistically strong but are scaffold-local (filter via `n_distinct_scaffolds`; see T7 in `counterfactual.py`) |
| Cleanlab "noisy" flag rate | small | **may be inflated** | activity cliffs look like label noise to confident-learning; do not auto-drop cliff compounds |

These ranges are *bounds-of-credibility*, not predictions. Any number
outside them on BRAF is a flag to investigate; any number inside them
should still be interpreted via `EXPLAINABILITY_SCOPE.md`.

### Mandatory practice for target-specific runs

1. Run `scripts/diagnose_dataset.py` first; refuse to interpret
   downstream results if `activity_cliff_fraction ≥ 0.10` was not
   handled with `scaffold_folds`.
2. Report **scaffold-split AUC**, not random-split AUC, as the headline
   number.
3. Add `--time-split` (when `document_year` is available) as a second
   eval lane — kinase data has 20+ years of medicinal-chemistry
   evolution; the past-vs-future split is the practical benchmark.
4. Annotate MMP rules with `n_distinct_scaffolds` and report
   `series-specific` vs `cross-series` separately.
5. Embed the `EXPLAINABILITY_SCOPE.md` boilerplate at the top of any
   HTML report produced from single-target data.

### References plugged in

The expected ranges above lean on these published anchors. Full text
in `EXPLAINABILITY_SCOPE.md`.

- Bharadwaj et al. 2024 (arXiv:2512.08077) — chemistry SAE on SMI-TED
- Auer et al. 2016 (PMC5198793) — MMP rules are scaffold-local
- van Tilborg et al. 2022 (PMC9749029) — activity cliffs degrade QSAR + SHAP
- PMC11032345 (2024) — kinase-class activity cliff density
- Cai et al. 2022 (PMC9208089) — TreeSHAP on kinases recovers ATP-binding pharmacophores

---

## Measured values — 2026-05-28/29 (ECFP4 path)

GPU-blocked tasks (Uni-Mol SAE on BRAF, CO-ADD reproducibility check)
are still pending RunPod stock recovery. See
`.omc/pods/20260529_qsar-braf-a40_stockout.yaml`.

### Cross-target comparison (single-target cancer kinases)

| Target | N | Active% | Scaffold OOF AUC | OOF AUPRC | SAE R² | n_series_local MMP | n_cross_series MMP |
|--------|---|---------|------------------|-----------|--------|-------------------|--------------------|
| CO-ADD PA (reference) | 24,120 | 2.9% | (stratified) 0.910 | 0.337 | 0.887 | 0/8 | 8/8 |
| BRAF | 5,751 | 40.9% | 0.909 | 0.878 | 0.872 | 1/8 | 7/8 |
| EGFR | 11,185 | 29.7% | 0.870 | 0.738 | 0.885 | 0/8 | 8/8 |
| JAK2 | 12,680 | 34.5% | 0.921 | 0.879 | 0.918 | 1/8 | 7/8 |

**Reading the table**:
- All three kinases hit AUC ≥ 0.87 with scaffold split → ECFP4 pipeline
  is "good enough" baseline regardless of which kinase.
- SAE R² 0.87–0.92 across the board → SAR-narrow vs diverse distinction
  is **not visible** on ECFP4 features (see PREMISES H2 — Uni-Mol
  measurement is the load-bearing test).
- MMP series-local fraction 0–12.5% with the default 8 SMIRKS — the
  preset transformations apply broadly. Probing kinase MMP "scaffold-
  bound" claim requires data-mined MMPs (deferred).

### Time-split AUC vs cutoff (BRAF, ECFP4 scaffold)

```
cutoff    n_train  n_test   AUC
2010      998      4753     0.548
2012      1572     4179     0.569
2014      2783     2968     0.603
2016      3647     2104     0.859
2018      5092     659      0.730  (small n_test — noisy)
2020      5448     303      0.875  (very small n_test)
```

→ **2014/2015 chemical-space discontinuity**. Pre-2014 training does
not predict post-2014 BRAF inhibitors at useful accuracy.
This is PREMISES H5 (the strongest single finding of the extension).

### Publication-bias mitigation effect (T-E2)

| Variant | N molecules | N active | active% |
|---------|-------------|----------|---------|
| BRAF default (pchembl-only) | 5,751 | 2,350 | 40.9% |
| BRAF + null-pchembl inactives | 8,671 | 2,350 | 27.1% |

Adding "null pchembl but standard_value present" rows as inactives
boosts the inactive pool by 2,920 molecules (+50%) and drops active%
to 27.1%. Whether this *improves* model honesty depends on whether
those rows are truly inactive — see
`scripts/download_chembl_target.py` per_molecule_with_null_inactives
docstring for the caveat.

### CO-ADD Uni-Mol reproducibility (2026-05-29, RTX 3090 — GH #2)

The EXPECTED_OUTPUTS Stage-6 A40 baseline (top of this doc) reproduced
on a fresh GPU 3 days later, different GPU class:

| Metric | A40 baseline (2026-05-26) | RTX 3090 reproduction | within tolerance? |
|---|---|---|---|
| OOF AUC (stratified, 5-fold) | 0.895 | **0.894** | ✓ (±0.01) |
| AUPRC | 0.256 | **0.259** | ✓ |
| SAE R²_median (latent=4096, ep=100) | 0.824 | **0.826** | ✓ (±0.05) |
| dead_ratio | 0.000 | 0.000 | ✓ |

**Reproducibility CONFIRMED.** The pipeline is GPU-class-independent
and stable across runs. This closes the regression-test concern: the
headline numbers are real, not run-specific artifacts.

### BRAF Uni-Mol comparison (2026-05-29, RTX A6000 + RTX 3090)

| Metric | ECFP4 (2048 bit) | Uni-Mol (512-d) | gap |
|---|---|---|---|
| Stratified OOF AUC | 0.928 | **0.832** | **-0.096 (Uni-Mol loses)** |
| Scaffold OOF AUC | 0.909 | **0.806** | **-0.103 (Uni-Mol loses)** |
| Scaffold OOF AUPRC | 0.878 | 0.729 | -0.149 |
| SAE R²_median | 0.872 (latent=2048, ep=30) | 0.852 (latent=4096, ep=100) | -0.020 |
| dead_ratio | 0.000 | 0.000 | 0 |

**Key finding**: Uni-Mol *underperforms* ECFP4 on BRAF — on **both**
stratified (-0.096) and scaffold (-0.103) splits (see PREMISES H6).
The README claim "Uni-Mol on full N beats ECFP4 here" was measured on
CO-ADD PA (phenotypic, diverse). On a target-specific kinase dataset
the relationship reverses, consistently across split protocols.
The two BRAF Uni-Mol scaffold runs (0.804 RTX A6000, 0.806 RTX 3090)
agree to 0.002 — another reproducibility datapoint.

### BRAF finetuned Uni-Mol (H7, 2026-05-29, RTX A6000)

Does **finetuning** (backprop through the transformer, not frozen
`get_repr`) close the frozen→ECFP4 gap? MolTrain scaffold 5-fold,
epochs=20, 2 seeds:

| Lane (BRAF, scaffold OOF AUC) | AUC |
|---|---|
| ECFP4 + XGB | **0.909** |
| finetuned Uni-Mol (epochs=20) | 0.887 ± 0.006 |
| frozen Uni-Mol + XGB | 0.806 |

Finetuning recovers **79%** of the frozen→ECFP4 gap (0.806→0.887,
+0.081 of 0.103) but does **not** reverse it — ECFP4 stays +0.022 ahead
(~4× the 0.006 seed spread). **Conclusion: the 3D foundation-model
advantage is not automatic on target-specific kinase QSAR.** Frozen
embeddings lose; finetuning mostly catches up at real GPU cost ($0.76
for this 2-seed run); the cheap ECFP4 baseline remains marginally best.
From-scratch (random-init) control is infra-blocked in `unimol_tools`
(always loads pretrained), so "pretraining helps" vs "NN helps" can't be
fully separated here — documented limitation. See PREMISES H7.

Per-descriptor SAE R² (Uni-Mol → linear-recover RDKit descriptors):

| Descriptor | R² |
|---|---|
| MolWt | 0.901 |
| HeavyAtomCount | 0.912 |
| TPSA | 0.882 |
| FractionCSP3 | 0.887 |
| NumRotatableBonds | 0.863 |
| NumHAcceptors | 0.840 |
| NumHDonors | 0.814 |
| MolLogP | 0.646 |
| NumAromaticRings | 0.619 |
| RingCount | 0.594 |

All 10 descriptors clear R² > 0.5; physicochemical bulk descriptors
near-perfect, count-of-rings descriptors moderate. The SAE recovers
descriptor structure regardless of input chemical diversity — H2
(diversity-dependent R²) is FALSIFIED on both representations.

**GPU cost**: $0.19 actual (RTX A6000 EU-SE-1, ~23 min wall clock,
$0.49/hr). Within approved $0.25 cap. Pod
`8nwcewfjld63l4`, deleted immediately after rsync.

### MMP grid (BRAF, inactive subset × series-local threshold)

| Subset | thr=2 | thr=3 | thr=5 |
|--------|-------|-------|-------|
| 500 | 0 | 0 | 0 |
| 1500 | 0 | 0 | 0 |
| 3401 (full) | 0 | 0 | 0 |

Zero series-local rules across all 9 cells. **The default SMIRKS
transformations are too generic to test scaffold-locality** — every
rule reaches > 5 distinct scaffolds. This invalidates PREMISES H4 for
the current transformation set (does not invalidate the Auer 2016
claim, which is about data-mined MMPs).
