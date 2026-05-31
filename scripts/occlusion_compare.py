"""Model-agnostic atom-occlusion — compare localization across targets + featurizers.

Atom occlusion: for a molecule the model calls active, mask each heavy atom
(-> C), re-featurize, re-predict p(active), and take Δp = p_full - p_masked per
atom. We ask two things, the same metrics as scripts/occlusion_braf.py:

  * CONCENTRATION (top-1 atom's share of total |Δp|): does a few atoms carry the
    call, or is it smeared? High = localized.
  * ACTIVE vs INACTIVE max|Δp| (Mann-Whitney, one-sided): do confident actives
    localize MORE than inactives (the model "found a driver")?

Unlike occlusion_braf.py (which loaded a finetuned Uni-Mol checkpoint), this runs
on CPU against either lane:
  --features ecfp4   ECFP4 bits + XGBoost   (the interpretable feature space)
  --features unimol  frozen Uni-Mol 512-d + XGBoost

so we can compare WHERE each model's reasoning sits — across BRAF (kinase, where
the finetuned-Uni-Mol occlusion came out NULL), TYMS (enzyme), ADAM10
(metalloprotease). NULL on one target + SIGNAL on another => NULL is a property of
that target's SAR, not of the method.

CAVEAT: the model is fit on ALL rows (this is an interpretability probe, not a
generalization test), so occluded molecules are in-sample. Same caveat the BRAF
run carried. The question here is "does the fitted model localize", not "does it
generalize".

Usage::

    PYTHONPATH=src python scripts/occlusion_compare.py \
        --csv data/processed/tyms.csv --features ecfp4 \
        --out reports/occlusion_tyms_ecfp4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import mannwhitneyu

from qsar_tutorial.featurizer import UniMolFeaturizer, featurize_ecfp4
from qsar_tutorial.model import build_classifier, fit_with_class_weight

RDLogger.DisableLog("rdApp.*")

# Reference: a textbook-pharmacophore case (sibling project FQ antibiotic occlusion)
# localized at a 3.5x concentration ratio. We compare our targets against that bar.
REF_FQ = {"fq_max_delta": 0.448, "nonfq_max_delta": 0.128, "ratio": 3.5}


def mask_atom_to_C(mol, atom_idx):
    """Mask one heavy atom to aliphatic carbon. Returns SMILES or None (no-op/invalid)."""
    rw = Chem.RWMol(mol)
    a = rw.GetAtomWithIdx(atom_idx)
    if a.GetAtomicNum() == 6 and not a.GetIsAromatic():
        return None  # already aliphatic C — masking is a no-op
    a.SetAtomicNum(6)
    a.SetNumExplicitHs(0)
    a.SetFormalCharge(0)
    a.SetIsAromatic(False)
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        return None


class Predictor:
    """Fit a featurizer+XGB lane on the full dataset; expose predict(smiles)->p(active)."""

    def __init__(self, features: str):
        self.features = features
        self._unimol = UniMolFeaturizer() if features == "unimol" else None
        self.model = None

    def _featurize(self, smiles: list[str]):
        if self.features == "ecfp4":
            return featurize_ecfp4(smiles)
        return self._unimol.featurize(smiles)

    def fit(self, smiles: list[str], y: np.ndarray) -> None:
        X, valid = self._featurize(smiles)
        self.model = build_classifier()
        fit_with_class_weight(self.model, X[valid], y[valid])

    def predict(self, smiles: list[str]) -> np.ndarray:
        X, valid = self._featurize(smiles)
        p = np.full(len(smiles), np.nan)
        if valid.any():
            p[valid] = self.model.predict_proba(X[valid])[:, 1]
        return p


def block(mols, tag):
    act = [m for m in mols if m["is_active"]]
    ina = [m for m in mols if not m["is_active"]]

    def mean(xs, k):
        return round(float(np.mean([x[k] for x in xs])), 4) if xs else None

    def mwu(a, b, k):
        av, bv = [x[k] for x in a], [x[k] for x in b]
        if len(av) < 3 or len(bv) < 3:
            return None
        try:
            return round(float(mannwhitneyu(av, bv, alternative="greater").pvalue), 4)
        except Exception:
            return None

    return {
        "tag": tag, "n_active": len(act), "n_inactive": len(ina),
        "active_mean_max_abs_delta": mean(act, "max_abs_delta"),
        "inactive_mean_max_abs_delta": mean(ina, "max_abs_delta"),
        "active_mean_top1_concentration": mean(act, "top1_concentration"),
        "inactive_mean_top1_concentration": mean(ina, "top1_concentration"),
        "mann_whitney_p_active_gt_inactive_maxdelta": mwu(act, ina, "max_abs_delta"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--features", choices=["ecfp4", "unimol"], default="ecfp4")
    p.add_argument("--n-per-class", type=int, default=50)
    p.add_argument("--conf-margin", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["canonical_smiles", "active"]).reset_index(drop=True)
    y = df["active"].to_numpy().astype(int)

    print(f"[fit] {args.features} on N={len(df)} ({y.mean():.1%} active) ...", flush=True)
    clf = Predictor(args.features)
    clf.fit(df["canonical_smiles"].tolist(), y)

    # pick confident-ish actives + inactives by the model's own p
    p_all = clf.predict(df["canonical_smiles"].tolist())
    df = df.assign(p_full=p_all).dropna(subset=["p_full"])
    act = df[df.active == 1].sample(n=min(args.n_per_class, int((df.active == 1).sum())), random_state=args.seed)
    ina = df[df.active == 0].sample(n=min(args.n_per_class, int((df.active == 0).sum())), random_state=args.seed)
    sel = pd.concat([act, ina]).reset_index(drop=True)
    print(f"[sel] {len(act)} actives + {len(ina)} inactives", flush=True)

    # enumerate masked variants
    full_records, masked_records = [], []
    for i, row in sel.iterrows():
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        if mol is None:
            continue
        full_records.append({"mid": int(i), "smiles": row.canonical_smiles,
                             "is_active": bool(row.active == 1), "n_atoms": mol.GetNumAtoms()})
        for a in range(mol.GetNumAtoms()):
            ms = mask_atom_to_C(mol, a)
            if ms is not None and ms != row.canonical_smiles:
                masked_records.append({"mid": int(i), "atom": int(a), "smiles": ms})
    print(f"[mask] full={len(full_records)} masked={len(masked_records)} — predicting ...", flush=True)

    all_smi = [r["smiles"] for r in full_records] + [r["smiles"] for r in masked_records]
    pred = clf.predict(all_smi)
    nf = len(full_records)
    pfull = {r["mid"]: float(pred[i]) for i, r in enumerate(full_records)}

    delta = {}
    for j, r in enumerate(masked_records):
        pm = pred[nf + j]
        if not np.isnan(pm):
            delta.setdefault(r["mid"], {})[r["atom"]] = pfull[r["mid"]] - float(pm)

    per_mol = []
    for r in full_records:
        d = delta.get(r["mid"], {})
        if not d:
            continue
        items = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)
        absvals = np.array([abs(v) for v in d.values()])
        per_mol.append({
            "mid": r["mid"], "is_active": r["is_active"], "p_full": round(pfull[r["mid"]], 3),
            "n_atoms": r["n_atoms"], "max_abs_delta": round(float(abs(items[0][1])), 4),
            "top1_concentration": round(float(abs(items[0][1]) / (absvals.sum() + 1e-9)), 3),
            "top5_atoms": [a for a, _ in items[:5]], "top5_delta": [round(v, 4) for _, v in items[:5]],
        })

    full_block = block(per_mol, "all")
    confident = [m for m in per_mol if abs(m["p_full"] - 0.5) >= args.conf_margin]
    conf_block = block(confident, f"confident(|p-0.5|>={args.conf_margin})")

    cp = conf_block["mann_whitney_p_active_gt_inactive_maxdelta"]
    cc_a = conf_block["active_mean_top1_concentration"]
    if conf_block["n_active"] < 3 or conf_block["n_inactive"] < 3:
        verdict = "INCONCLUSIVE: too few confident molecules"
    elif cp is not None and cp < 0.05:
        verdict = "SIGNAL: confident actives localize more than inactives"
    else:
        verdict = "NULL: occlusion does not separate actives from inactives (diffuse attribution)"

    report = {
        "summary": {
            "csv": args.csv, "features": args.features,
            "n_molecules": int(len(df)), "active_fraction": round(float(y.mean()), 4),
            "caveat": "model fit on ALL rows; occluded molecules in-sample (interpretability probe, not generalization)",
            "conf_margin": args.conf_margin,
            "full": full_block, "confident": conf_block,
            "reference_FQ_contrast": REF_FQ,
            "verdict": verdict,
        },
        "per_molecule": per_mol,
    }
    out = args.out or args.csv.replace("data/processed/", "reports/occlusion_").replace(".csv", f"_{args.features}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {out}\n  verdict: {verdict}")
    print(f"  confident: act_conc={cc_a} ina_conc={conf_block['inactive_mean_top1_concentration']} "
          f"act_maxΔ={conf_block['active_mean_max_abs_delta']} ina_maxΔ={conf_block['inactive_mean_max_abs_delta']} "
          f"MWp={cp}")


if __name__ == "__main__":
    main()
