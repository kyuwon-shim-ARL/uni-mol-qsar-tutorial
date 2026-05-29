#!/usr/bin/env python3
"""Measure SAE feature monosemanticity — the load-bearing SAE quality metric
that PREMISES H2's open question asks for (descriptor R² is a coarser proxy).

A *monosemantic* latent fires for one interpretable concept; a *polysemantic*
latent mixes several. We operationalize this on the SAE code Z (N × latent)
against K interpretable RDKit descriptors D (N × K):

  For each ACTIVE latent j (fires on > 1% of molecules):
    - Ridge-regress each descriptor d_k on z_j alone → r2_jk
    - selectivity_j = max_k r2_jk                  (best single descriptor)
    - runner_up_j   = 2nd-highest r2_jk
    - margin_j      = selectivity_j - runner_up_j  (how cleanly it wins)
    - monosemantic  = selectivity_j >= 0.5 AND margin_j >= 0.2

This is deliberately conservative: a latent is "monosemantic w.r.t. our
descriptor panel" only if one descriptor explains it well AND clearly beats
the second. It does NOT claim the latent has no other meaning outside the
10-descriptor panel — that is the documented limitation.

Run on ECFP4 SAE (CPU) by default; pass --features unimol on a GPU box for
the dense-embedding case.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RDLogger.DisableLog("rdApp.*")

from qsar_tutorial.data import load_coadd_pa
from qsar_tutorial.featurizer import featurize_ecfp4
from qsar_tutorial.sae import train_sae

DESCS = ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
         "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
         "HeavyAtomCount", "RingCount"]


def compute_descs(smis):
    out = np.zeros((len(smis), len(DESCS)), dtype=np.float32)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        for j, n in enumerate(DESCS):
            try:
                out[i, j] = float(getattr(Descriptors, n)(m))
            except Exception:
                pass
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--features", choices=["ecfp4", "unimol"], default="ecfp4")
    p.add_argument("--sae-latent", type=int, default=2048)
    p.add_argument("--sae-epochs", type=int, default=30)
    p.add_argument("--fire-threshold", type=float, default=0.01,
                   help="latent counts as active if it fires on >= this fraction")
    p.add_argument("--sel-min", type=float, default=0.5,
                   help="monosemantic if best-descriptor R2 >= this")
    p.add_argument("--margin-min", type=float, default=0.2,
                   help="and best - runner-up R2 >= this")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ds = load_coadd_pa(args.csv)
    if args.features == "ecfp4":
        X, valid = featurize_ecfp4(list(ds.smiles), n_bits=2048)
    else:
        from qsar_tutorial.featurizer import UniMolFeaturizer
        X, valid = UniMolFeaturizer().featurize(list(ds.smiles))
    X = X[valid].astype(np.float32)
    smi = ds.smiles[valid]
    print(f"Loaded {len(smi)} mols, X={X.shape} ({args.features})", flush=True)

    sae = train_sae(X, latent_dim=args.sae_latent, epochs=args.sae_epochs, verbose=False)
    Z = sae.encode(X)  # N x latent
    D = compute_descs(smi)
    # standardize descriptors
    D = (D - D.mean(0)) / (D.std(0) + 1e-8)

    fire_rate = (Z > 1e-6).mean(0)
    active = np.where(fire_rate >= args.fire_threshold)[0]
    print(f"Active latents: {len(active)}/{Z.shape[1]} (fire >= {args.fire_threshold:.0%})", flush=True)

    rng = np.random.default_rng(42)
    # subsample molecules for speed if huge
    n = len(smi)
    idx = rng.choice(n, size=min(n, 4000), replace=False) if n > 4000 else np.arange(n)
    Zs, Ds = Z[idx], D[idx]

    records = []
    for j in active:
        zj = Zs[:, j:j+1]
        r2s = []
        for k in range(len(DESCS)):
            # single-feature ridge, 3-fold CV R2
            try:
                sc = cross_val_score(Ridge(alpha=1.0), zj, Ds[:, k], cv=3,
                                     scoring="r2")
                r2s.append(float(max(0.0, sc.mean())))
            except Exception:
                r2s.append(0.0)
        order = np.argsort(r2s)[::-1]
        sel = r2s[order[0]]
        runner = r2s[order[1]] if len(order) > 1 else 0.0
        margin = sel - runner
        mono = sel >= args.sel_min and margin >= args.margin_min
        records.append({
            "latent": int(j),
            "best_descriptor": DESCS[order[0]],
            "selectivity": round(sel, 3),
            "runner_up": round(runner, 3),
            "margin": round(margin, 3),
            "monosemantic": bool(mono),
        })

    n_mono = sum(r["monosemantic"] for r in records)
    n_active = len(records)
    mono_frac = n_mono / max(n_active, 1)
    # which descriptors "own" monosemantic latents
    from collections import Counter
    owner = Counter(r["best_descriptor"] for r in records if r["monosemantic"])

    report = {
        "csv": args.csv, "features": args.features,
        "sae_latent": args.sae_latent, "sae_epochs": args.sae_epochs,
        "n_active_latents": n_active,
        "n_monosemantic": n_mono,
        "monosemantic_fraction": round(mono_frac, 3),
        "criteria": {"sel_min": args.sel_min, "margin_min": args.margin_min,
                     "fire_threshold": args.fire_threshold},
        "monosemantic_by_descriptor": dict(owner),
        "latents": sorted(records, key=lambda r: r["selectivity"], reverse=True)[:50],
    }
    out = args.out or f"reports/20260529_sae_monosem_{args.features}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))

    print(f"\n=== Monosemanticity ({args.features}) ===", flush=True)
    print(f"  active latents: {n_active}", flush=True)
    print(f"  monosemantic:   {n_mono} ({mono_frac:.1%})  "
          f"[sel>={args.sel_min}, margin>={args.margin_min}]", flush=True)
    print(f"  owned by: {dict(owner)}", flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
