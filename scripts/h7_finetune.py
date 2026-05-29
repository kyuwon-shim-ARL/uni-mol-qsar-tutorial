"""H7: does finetuning Uni-Mol close the gap frozen embeddings lost?

Lanes (MolTrain split='scaffold' k-fold):
  - finetuned : MolTrain on pretrained unimolv1 backbone
Compared (offline, from prior measured runs) against:
  - frozen-UniMol+XGB scaffold (BRAF 0.806), ECFP4+XGB scaffold (BRAF 0.909)

NOTE: a from-scratch (random-init) control was attempted but unimol_tools
MolTrain always loads the pretrained backbone (pretrained_model_path="" errors
needing dict.txt). The control is infra-blocked and documented as a limitation:
this run shows "finetuning beats frozen?" but cannot fully separate
"pretraining helps" from "end-to-end NN helps".

Usage: python h7_finetune.py <csv> <tag> <epochs> [seeds...]
"""
import sys, json, glob, os, time
import numpy as np
import joblib
from pathlib import Path

csv = sys.argv[1]; tag = sys.argv[2]
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
seeds = [int(s) for s in sys.argv[4:]] or [42, 1]

import pandas as pd
df = pd.read_csv(csv).dropna(subset=["canonical_smiles", "active"])
df = df.rename(columns={"canonical_smiles": "SMILES", "active": "TARGET"})
work = Path("h7_work"); work.mkdir(exist_ok=True)
data_csv = work / f"{tag}_data.csv"
df[["SMILES", "TARGET"]].to_csv(data_csv, index=False)
print(f"[{tag}] N={len(df)} active={df.TARGET.mean():.1%} epochs={epochs} seeds={seeds}", flush=True)

from unimol_tools import MolTrain

def read_metric(save_path):
    mr = os.path.join(save_path, "metric.result")
    try:
        d = joblib.load(mr)
        return float(d.get("auc", d.get("auroc"))), float(d.get("auprc"))
    except Exception as e:
        print(f"  [metric read fail {mr}] {e}", flush=True)
        return None, None

def run_lane(lane, seed):
    sp = str(work / f"{tag}_{lane}_s{seed}")
    kw = dict(task="classification", epochs=epochs, learning_rate=1e-4,
              batch_size=32, early_stopping=5, metrics="auc",
              split="scaffold", kfold=5, save_path=sp,
              smiles_col="SMILES", target_cols=["TARGET"],
              model_name="unimolv1", model_size="84m", seed=seed)
    t0 = time.time()
    try:
        MolTrain(**kw).fit(data=str(data_csv))
    except TypeError:
        kw.pop("seed", None); MolTrain(**kw).fit(data=str(data_csv))
    auc, auprc = read_metric(sp)
    print(f"  [{lane} s{seed}] AUC={auc} AUPRC={auprc} ({time.time()-t0:.0f}s)", flush=True)
    return auc, auprc

results = {"csv": csv, "tag": tag, "epochs": epochs, "lanes": {}}
aucs, auprcs = [], []
for seed in seeds:
    try:
        auc, auprc = run_lane("finetuned", seed)
        if auc is not None: aucs.append(auc); auprcs.append(auprc)
    except Exception as e:
        print(f"  [finetuned s{seed} FAILED] {e}", flush=True)
if aucs:
    results["lanes"]["finetuned"] = {
        "seeds": seeds[:len(aucs)], "aucs": [round(a,4) for a in aucs],
        "auprcs": [round(a,4) for a in auprcs],
        "mean_auc": round(float(np.mean(aucs)),4), "std_auc": round(float(np.std(aucs)),4),
    }
    print(f"[{tag}:finetuned] mean AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f} (n={len(aucs)})", flush=True)
results["from_scratch_control"] = "infra-blocked: unimol_tools MolTrain always loads pretrained backbone"
results["reference_lanes"] = {"frozen_unimol_xgb_scaffold": 0.806, "ecfp4_xgb_scaffold": 0.909}

out = f"reports_gpu/h7_{tag}.json"
Path("reports_gpu").mkdir(exist_ok=True)
Path(out).write_text(json.dumps(results, indent=2))
print(f"-> {out}", flush=True); print(f"H7_DONE_{tag}", flush=True)
