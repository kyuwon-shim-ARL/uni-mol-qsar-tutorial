"""Finetuning lane for H7 — does finetuning Uni-Mol recover the loss that
frozen embeddings suffer (H6)?

This is the third lane alongside ECFP4+XGBoost and frozen-Uni-Mol+XGBoost
(both produced by ``examples/run_split_comparison.py``). It finetunes the
Uni-Mol backbone end-to-end (backprop through the transformer, not just
``get_repr``) on the target's labels.

Design (PREMISES.md#H7, tcrit-hardened):

* **Scaffold parity** — the SAME Bemis-Murcko scaffold folds the XGBoost
  lanes use. We build fold ids with ``qsar_tutorial.data.scaffold_folds``
  and feed them to MolTrain via ``split='select'`` (it reads fold ids
  0..k-1 from a column). The internal Splitter seed is fixed, so folds are
  identical across seeds and across lanes.
* **From-scratch control** (``--modes scratch``) — same architecture, no
  pretrained weights (``load_pretrained_weights`` monkeypatched to a no-op).
  A finetuned win over THIS proves pretraining helps, not merely that an
  end-to-end NN beats XGBoost.
* **Variance** — multi-seed (``--seeds``). ``torch.manual_seed`` varies
  weight init + training stochasticity while folds stay fixed. The frozen
  vs ECFP4 gap to beat on TYMS is 0.069 with fold_std ~0.07, so a real win
  must clear the seed spread.
* **HP budget is logged** — epochs / lr / batch / freeze are recorded in
  the output JSON so a loss from bad HPs is not mis-read as refuting H7.

Usage::

    PYTHONPATH=src python scripts/finetune_unimol.py \
        --csv data/processed/tyms.csv \
        --modes finetune scratch --seeds 0 1 2 \
        --epochs 30 --out reports/20260529_tyms_h7_finetune.json

    # fast correctness check (1 seed, 1 fold-worth, 2 epochs):
    PYTHONPATH=src python scripts/finetune_unimol.py --csv data/processed/tyms.csv --smoke
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from qsar_tutorial.data import load_coadd_pa, scaffold_folds


def build_fold_ids(ds, n_splits: int, seed: int) -> np.ndarray:
    """Per-row scaffold fold id (0..n_splits-1) — the validation fold of each row."""
    folds = scaffold_folds(ds, n_splits=n_splits, seed=seed)
    fold_id = np.full(len(ds.y), -1, dtype=int)
    for k, (_, val_idx) in enumerate(folds):
        fold_id[val_idx] = k
    assert (fold_id >= 0).all(), "every row must land in exactly one validation fold"
    return fold_id


def _positive_scores(cv_pred: np.ndarray) -> np.ndarray:
    """MolTrain classification cv_pred is (N, n_classes) softmax probs (or (N,))."""
    arr = np.asarray(cv_pred)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.ravel()


def run_one(
    df: pd.DataFrame,
    *,
    mode: str,
    seed: int,
    n_splits: int,
    epochs: int,
    lr: float,
    batch_size: int,
    freeze_layers,
    save_root: str,
) -> dict:
    """One MolTrain run; returns scaffold-OOF AUC/AUPRC + per-fold AUCs."""
    import torch
    import unimol_tools.models.unimol as um
    from unimol_tools import MolTrain

    torch.manual_seed(seed)
    np.random.seed(seed)

    # from-scratch control: disable pretrained weight loading for this run
    original_loader = um.UniMolModel.load_pretrained_weights
    if mode == "scratch":
        um.UniMolModel.load_pretrained_weights = lambda self, *a, **k: None

    try:
        save_path = os.path.join(save_root, f"{mode}_seed{seed}")
        clf = MolTrain(
            task="classification",
            data_type="molecule",
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            metrics="auc",
            split="select",          # use precomputed fold ids -> scaffold parity
            split_group_col="fold",
            kfold=n_splits,
            save_path=save_path,
            smiles_col="canonical_smiles",
            target_cols="active",
            freeze_layers=freeze_layers,
            use_gpu=torch.cuda.is_available(),
            remove_hs=False,
            model_name="unimolv1",
            model_size="84m",
            # MolTrain's Trainer calls set_seed(config.seed) (default 42), which
            # overrides any torch.manual_seed set here. Propagate the real seed
            # via params so multi-seed actually varies init + training (the
            # split_seed stays fixed → identical scaffold folds across seeds).
            params={"seed": seed},
        )
        clf.fit(df)
        scores = _positive_scores(clf.cv_pred)
        y_true = np.asarray(clf.data["target"]).ravel().astype(int)
    finally:
        um.UniMolModel.load_pretrained_weights = original_loader

    fold = df["fold"].to_numpy()
    per_fold = []
    for k in range(n_splits):
        m = fold == k
        if m.sum() > 0 and len(np.unique(y_true[m])) > 1:
            per_fold.append(round(float(roc_auc_score(y_true[m], scores[m])), 4))
    return {
        "mode": mode,
        "seed": seed,
        "oof_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "oof_auprc": round(float(average_precision_score(y_true, scores)), 4),
        "per_fold_auc": per_fold,
        "fold_std": round(float(np.std(per_fold)), 4) if per_fold else None,
    }


def summarize(runs: list[dict]) -> dict:
    """Aggregate per-mode across seeds: mean / std / 95% CI of the OOF AUC."""
    out = {}
    for mode in sorted({r["mode"] for r in runs}):
        aucs = np.array([r["oof_auc"] for r in runs if r["mode"] == mode], dtype=float)
        n = len(aucs)
        mean = float(aucs.mean())
        std = float(aucs.std(ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        out[mode] = {
            "n_seeds": n,
            "mean_oof_auc": round(mean, 4),
            "std": round(std, 4),
            "ci95": [round(mean - 1.96 * sem, 4), round(mean + 1.96 * sem, 4)],
            "seeds": [r["oof_auc"] for r in runs if r["mode"] == mode],
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="processed CSV: canonical_smiles, active")
    p.add_argument("--modes", nargs="+", default=["finetune", "scratch"],
                   choices=["finetune", "scratch"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--freeze-layers", default=None,
                   help="MolTrain freeze_layers (layer-name prefix); default None = full finetune")
    p.add_argument("--fold-seed", type=int, default=42,
                   help="seed for scaffold fold assignment (kept fixed for parity)")
    p.add_argument("--save-root", default=None, help="MolTrain checkpoint dir (default: tmp under reports/)")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--smoke", action="store_true",
                   help="fast correctness check: 1 seed, 2 epochs, finetune only")
    args = p.parse_args()

    if args.smoke:
        args.modes = ["finetune"]
        args.seeds = [0]
        args.epochs = 2

    ds = load_coadd_pa(args.csv)
    fold_id = build_fold_ids(ds, n_splits=args.kfold, seed=args.fold_seed)
    df = pd.DataFrame({"canonical_smiles": ds.smiles, "active": ds.y.astype(int), "fold": fold_id})

    save_root = args.save_root or os.path.join("reports", "_finetune_ckpt")
    Path(save_root).mkdir(parents=True, exist_ok=True)

    print(f"[finetune_unimol] csv={args.csv} N={len(df)} active={df['active'].mean():.1%} "
          f"folds={args.kfold} modes={args.modes} seeds={args.seeds} epochs={args.epochs}")

    runs = []
    for mode in args.modes:
        for seed in args.seeds:
            print(f"  -> mode={mode} seed={seed} ...", flush=True)
            r = run_one(
                df, mode=mode, seed=seed, n_splits=args.kfold, epochs=args.epochs,
                lr=args.lr, batch_size=args.batch_size, freeze_layers=args.freeze_layers,
                save_root=save_root,
            )
            print(f"     oof_auc={r['oof_auc']} per_fold={r['per_fold_auc']}", flush=True)
            runs.append(r)

    report = {
        "csv": args.csv,
        "n_molecules": int(len(df)),
        "active_fraction": round(float(df["active"].mean()), 4),
        "kfold": args.kfold,
        "fold_seed": args.fold_seed,
        "hp": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
               "freeze_layers": args.freeze_layers},
        "runs": runs,
        "summary": summarize(runs),
        "baseline_to_beat": {"ecfp4_scaffold_auc": 0.800, "frozen_unimol_scaffold_auc": 0.731,
                             "note": "from reports/20260529_split-compare_{ecfp4,unimol}.json (tyms)"},
    }

    out = args.out or args.csv.replace("data/processed/", "reports/").replace(".csv", "_h7_finetune.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[finetune_unimol] wrote {out}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
