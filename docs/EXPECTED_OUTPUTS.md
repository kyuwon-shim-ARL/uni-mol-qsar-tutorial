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
