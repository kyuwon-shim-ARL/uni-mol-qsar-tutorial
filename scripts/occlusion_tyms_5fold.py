#!/usr/bin/env python3
"""Atom-level occlusion on the tutorial's finetuned Uni-Mol TYMS classifier.

Parallel to the sibling project's e189v3 occlusion (which localized the FQ
pharmacophore at 72.7%). Here there is no single textbook pharmacophore, so we
test the *model-agnostic* interpretability signal directly:

  For each molecule, mask each heavy atom (→ C), re-predict p(active), and take
  Δp = p_full − p_masked per atom. A model that bases its call on specific
  substructure should show (a) CONCENTRATED attribution (a few atoms dominate)
  and (b) a LARGER max-Δp on actives than on inactives (it "found" the driver).

This is the dictionary-free interpretability path (no SAE latent naming).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CKPT = "reports/_finetune_ckpt/finetune_seed0"
CSV = "data/processed/tyms.csv"
OUT = "reports/occlusion_tyms_5fold.json"
N_PER_CLASS = 50
CONF_MARGIN = 0.2   # "confident" = |p_full - 0.5| >= this (occlusion only interpretable when model has a strong opinion)
SEED = 42


def mask_atom_to_C(mol, atom_idx):
    rw = Chem.RWMol(mol)
    a = rw.GetAtomWithIdx(atom_idx)
    if a.GetAtomicNum() == 6 and not a.GetIsAromatic():
        return None
    a.SetAtomicNum(6); a.SetNumExplicitHs(0); a.SetFormalCharge(0); a.SetIsAromatic(False)
    try:
        Chem.SanitizeMol(rw); return Chem.MolToSmiles(rw)
    except Exception:
        return None


def main():
    df = pd.read_csv(CSV)
    act = df[df.active == 1].sample(n=min(N_PER_CLASS, int((df.active == 1).sum())), random_state=SEED)
    ina = df[df.active == 0].sample(n=min(N_PER_CLASS, int((df.active == 0).sum())), random_state=SEED)
    sel = pd.concat([act, ina]).reset_index(drop=True)
    sel["is_active"] = (sel.active == 1)
    print(f"[sel] {len(act)} actives + {len(ina)} inactives", flush=True)

    full_records, masked_records = [], []
    for i, row in sel.iterrows():
        smi = row.canonical_smiles
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        full_records.append({"mid": int(i), "smiles": smi, "is_active": bool(row.is_active),
                             "n_atoms": mol.GetNumAtoms()})
        for a in range(mol.GetNumAtoms()):
            ms = mask_atom_to_C(mol, a)
            if ms is not None and ms != smi:
                masked_records.append({"mid": int(i), "atom": int(a), "smiles": ms})
    print(f"[mask] full={len(full_records)}, masked={len(masked_records)}", flush=True)

    from unimol_tools import MolPredict
    all_smi = [r["smiles"] for r in full_records] + [r["smiles"] for r in masked_records]
    mp = MolPredict(load_model=CKPT)
    preds = np.asarray(mp.predict(pd.DataFrame({"canonical_smiles": all_smi})))
    # binary (num_classes=1) → p(active)
    if preds.ndim == 2:
        p = preds[:, 1] if preds.shape[1] == 2 else preds[:, 0]
    else:
        p = preds.ravel()
    print(f"[predict] preds shape {preds.shape} → p(active) vec {p.shape}", flush=True)

    nf = len(full_records)
    pfull = {r["mid"]: float(p[i]) for i, r in enumerate(full_records)}
    delta = {}
    for j, r in enumerate(masked_records):
        delta.setdefault(r["mid"], {})[r["atom"]] = pfull[r["mid"]] - float(p[nf + j])

    per_mol = []
    for r in full_records:
        d = delta.get(r["mid"], {})
        if not d:
            continue
        items = sorted(d.items(), key=lambda x: abs(x[1]), reverse=True)
        top = items[:5]
        absvals = np.array([abs(v) for v in d.values()])
        concentration = float(abs(top[0][1]) / (absvals.sum() + 1e-9))  # top-1 share of total |Δp|
        per_mol.append({"mid": r["mid"], "is_active": r["is_active"], "p_full": round(pfull[r["mid"]], 3),
                        "n_atoms": r["n_atoms"], "max_delta": round(float(max(v for _, v in top)), 4),
                        "max_abs_delta": round(float(abs(top[0][1])), 4),
                        "top5_atoms": [a for a, _ in top], "top5_delta": [round(v, 4) for _, v in top],
                        "top1_concentration": round(concentration, 3)})

    from scipy.stats import mannwhitneyu
    def mean(xs, k): return round(float(np.mean([x[k] for x in xs])), 4) if xs else None
    def mwu(act, ina, k):
        a = [x[k] for x in act]; b = [x[k] for x in ina]
        if len(a) < 3 or len(b) < 3:
            return None
        try:
            return round(float(mannwhitneyu(a, b, alternative="greater").pvalue), 4)
        except Exception:
            return None

    def block(mols, tag):
        act = [m for m in mols if m["is_active"]]
        ina = [m for m in mols if not m["is_active"]]
        return {
            "tag": tag, "n_active": len(act), "n_inactive": len(ina),
            "active_mean_max_abs_delta": mean(act, "max_abs_delta"),
            "inactive_mean_max_abs_delta": mean(ina, "max_abs_delta"),
            "active_mean_top1_concentration": mean(act, "top1_concentration"),
            "inactive_mean_top1_concentration": mean(ina, "top1_concentration"),
            # one-sided test: are actives' max|Δp| GREATER than inactives'? (signal if p<0.05)
            "mann_whitney_p_active_gt_inactive_maxdelta": mwu(act, ina, "max_abs_delta"),
        }

    full_block = block(per_mol, "all")
    confident = [m for m in per_mol if abs(m["p_full"] - 0.5) >= CONF_MARGIN]
    conf_block = block(confident, f"confident(|p-0.5|>={CONF_MARGIN})")

    # Verdict
    cp = conf_block["mann_whitney_p_active_gt_inactive_maxdelta"]
    if conf_block["n_active"] < 3 or conf_block["n_inactive"] < 3:
        verdict = "INCONCLUSIVE: too few confident molecules (2-fold ensemble under-confident)"
    elif cp is not None and cp < 0.05:
        verdict = "SIGNAL: confident actives localize more than inactives — earlier flat result was low-confidence dilution"
    else:
        verdict = "NULL-CONFIRMED: even on confident molecules, occlusion does NOT localize actives vs inactives on this BRAF model (2-fold caveat)"

    summary = {
        "model": CKPT, "note": "2-of-5 fold ensemble (partial); molecules may overlap finetune train set — see caveat",
        "conf_margin": CONF_MARGIN,
        "full": full_block, "confident": conf_block,
        "reference_FQ_contrast": {"fq_max_delta": 0.448, "nonfq_max_delta": 0.128, "ratio": 3.5,
                                   "source": "/tmp/orig_occl_repro/occlusion_full.json (reproduced)"},
        "verdict": verdict,
    }
    rep = {"summary": summary, "per_molecule": sorted(per_mol, key=lambda m: m["max_abs_delta"], reverse=True)}
    Path(OUT).write_text(json.dumps(rep, indent=2))
    print("\n=== BRAF occlusion (finetuned Uni-Mol, 2-fold) ===", flush=True)
    for b in (full_block, conf_block):
        print(f"  [{b['tag']}] act max|Δp|={b['active_mean_max_abs_delta']} vs ina {b['inactive_mean_max_abs_delta']} "
              f"| act conc={b['active_mean_top1_concentration']} vs ina {b['inactive_mean_top1_concentration']} "
              f"| MWU p(act>ina)={b['mann_whitney_p_active_gt_inactive_maxdelta']} (n {b['n_active']}/{b['n_inactive']})", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
