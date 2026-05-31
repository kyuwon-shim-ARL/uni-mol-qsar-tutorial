#!/usr/bin/env python3
"""Route A — atom occlusion on the STRONG ECFP4+XGBoost TYMS model (OOF, GPU-free).

Counterpart to scripts/occlusion_braf.py (which ran on a weak 2-fold/2-epoch
Uni-Mol model and was flat). Here we use the well-trained ECFP4+XGB model
(scaffold OOF AUC ~0.909) so a flat result can't be blamed on a weak model.

OOF discipline: 5 scaffold folds; each molecule is predicted by the fold model
that did NOT train on it (no leakage). Occlusion: mask each heavy atom (→ C),
recompute ECFP4 of the masked molecule, predict with the molecule's OOF model,
Δp = p_full − p_masked. Same metrics as the Uni-Mol run for a direct contrast:
active-vs-inactive max|Δp| + top-1 concentration + Mann-Whitney, full + a
model-confident subset (|p−0.5|≥0.2).

Hypothesis (user): Uni-Mol may fail where ECFP4 succeeds — if ECFP4 occlusion
shows an active-vs-inactive contrast that the Uni-Mol run lacked, the BRAF null
was (partly) a weak-model artifact, not an intrinsic "occlusion can't work here".
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RDLogger.DisableLog("rdApp.*")

from qsar_tutorial.data import load_coadd_pa, scaffold_folds
from qsar_tutorial.featurizer import featurize_ecfp4
from qsar_tutorial.model import build_classifier

CSV = "data/processed/tyms.csv"
OUT = "reports/occlusion_tyms_ecfp4.json"
N_PER_CLASS = 50
CONF_MARGIN = 0.2
N_BITS = 2048
SEED = 42


def mask_atom_to_C(mol, idx):
    rw = Chem.RWMol(mol)
    a = rw.GetAtomWithIdx(idx)
    if a.GetAtomicNum() == 6 and not a.GetIsAromatic():
        return None
    a.SetAtomicNum(6); a.SetNumExplicitHs(0); a.SetFormalCharge(0); a.SetIsAromatic(False)
    try:
        Chem.SanitizeMol(rw); return Chem.MolToSmiles(rw)
    except Exception:
        return None


def main():
    ds = load_coadd_pa(CSV)
    smiles = list(ds.smiles); y = np.asarray(ds.y)
    X, valid = featurize_ecfp4(smiles, n_bits=N_BITS)
    folds = scaffold_folds(ds, n_splits=5, seed=SEED)

    # fold id per row (validation fold) + OOF fold model trained on the rest
    fold_id = np.full(len(y), -1)
    fold_models = {}
    oof_p = np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(folds):
        fold_id[va] = k
        clf = build_classifier()
        clf.fit(X[tr], y[tr])
        fold_models[k] = clf
        oof_p[va] = clf.predict_proba(X[va])[:, 1]
    from sklearn.metrics import roc_auc_score
    oof_auc = float(roc_auc_score(y, oof_p))
    print(f"[model] scaffold OOF AUC = {oof_auc:.3f} (strong model — sanity)", flush=True)

    rng = np.random.default_rng(SEED)
    act_idx = np.where(y == 1)[0]; ina_idx = np.where(y == 0)[0]
    sel = np.concatenate([rng.choice(act_idx, min(N_PER_CLASS, len(act_idx)), replace=False),
                          rng.choice(ina_idx, min(N_PER_CLASS, len(ina_idx)), replace=False)])

    per_mol = []
    for mi in sel:
        smi = smiles[mi]; mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        k = int(fold_id[mi]); clf = fold_models[k]
        p_full = float(oof_p[mi])
        masked_smi, atom_ids = [], []
        for a in range(mol.GetNumAtoms()):
            ms = mask_atom_to_C(mol, a)
            if ms is not None and ms != smi:
                masked_smi.append(ms); atom_ids.append(a)
        if not masked_smi:
            continue
        Xm, vm = featurize_ecfp4(masked_smi, n_bits=N_BITS)
        pm = clf.predict_proba(Xm)[:, 1]
        deltas = {a: p_full - float(pm[j]) for j, a in enumerate(atom_ids) if vm[j]}
        if not deltas:
            continue
        items = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        absvals = np.array([abs(v) for v in deltas.values()])
        per_mol.append({"mid": int(mi), "is_active": bool(y[mi] == 1), "p_full": round(p_full, 3),
                        "n_atoms": mol.GetNumAtoms(), "max_abs_delta": round(float(abs(items[0][1])), 4),
                        "top5_atoms": [a for a, _ in items[:5]], "top5_delta": [round(v, 4) for _, v in items[:5]],
                        "top1_concentration": round(float(abs(items[0][1]) / (absvals.sum() + 1e-9)), 3)})

    def mean(xs, kk): return round(float(np.mean([x[kk] for x in xs])), 4) if xs else None
    def mwu(a, b, kk):
        av = [x[kk] for x in a]; bv = [x[kk] for x in b]
        if len(av) < 3 or len(bv) < 3: return None
        try: return round(float(mannwhitneyu(av, bv, alternative="greater").pvalue), 4)
        except Exception: return None

    def block(mols, tag):
        a = [m for m in mols if m["is_active"]]; b = [m for m in mols if not m["is_active"]]
        return {"tag": tag, "n_active": len(a), "n_inactive": len(b),
                "active_mean_max_abs_delta": mean(a, "max_abs_delta"),
                "inactive_mean_max_abs_delta": mean(b, "max_abs_delta"),
                "active_mean_top1_concentration": mean(a, "top1_concentration"),
                "inactive_mean_top1_concentration": mean(b, "top1_concentration"),
                "mann_whitney_p_active_gt_inactive_maxdelta": mwu(a, b, "max_abs_delta")}

    full_b = block(per_mol, "all")
    conf = [m for m in per_mol if abs(m["p_full"] - 0.5) >= CONF_MARGIN]
    conf_b = block(conf, f"confident(|p-0.5|>={CONF_MARGIN})")
    cp = conf_b["mann_whitney_p_active_gt_inactive_maxdelta"]
    if conf_b["n_active"] < 3 or conf_b["n_inactive"] < 3:
        verdict = "INCONCLUSIVE: too few confident molecules"
    elif cp is not None and cp < 0.05:
        verdict = "SIGNAL: confident actives localize MORE than inactives on the strong ECFP4 model — occlusion works here (Uni-Mol 2-fold null was a weak-model artifact)"
    else:
        verdict = "NULL: even on the strong ECFP4 model, occlusion shows no active-vs-inactive localization contrast — the BRAF null is model-independent (no single localizable driver)"

    rep = {"model": "ECFP4(2048)+XGBoost, scaffold 5-fold OOF", "oof_auc": round(oof_auc, 3),
           "conf_margin": CONF_MARGIN, "full": full_b, "confident": conf_b,
           "reference_FQ_contrast": {"fq_max_delta": 0.448, "nonfq_max_delta": 0.128, "ratio": 3.5},
           "uni_mol_2fold_comparison": {"note": "Uni-Mol 2-fold run (occlusion_braf.json) was flat: active 0.105 vs inactive 0.110, MWU p=0.52 on confident subset"},
           "verdict": verdict}
    Path(OUT).write_text(json.dumps({"summary": rep, "per_molecule": sorted(per_mol, key=lambda m: m["max_abs_delta"], reverse=True)}, indent=2))
    print("\n=== BRAF occlusion on STRONG ECFP4+XGB (OOF AUC %.3f) ===" % oof_auc, flush=True)
    for b in (full_b, conf_b):
        print(f"  [{b['tag']}] act max|Δp|={b['active_mean_max_abs_delta']} vs ina {b['inactive_mean_max_abs_delta']} "
              f"| conc {b['active_mean_top1_concentration']} vs {b['inactive_mean_top1_concentration']} "
              f"| MWU p={b['mann_whitney_p_active_gt_inactive_maxdelta']} (n {b['n_active']}/{b['n_inactive']})", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
