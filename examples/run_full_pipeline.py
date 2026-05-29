"""End-to-end run on CO-ADD PA binary data.

Usage:
    python examples/run_full_pipeline.py \
        --csv data/processed/coadd_pa.csv \
        --features ecfp4 \
        --out reports/coadd_pa_report.html

To switch to the Uni-Mol path, pass --features unimol (requires unimol-tools
installed; downloads weights from HuggingFace on first use; GPU recommended).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.metrics import average_precision_score, roc_auc_score

from qsar_tutorial.counterfactual import scan
from qsar_tutorial.data import (
    Dataset, load_coadd_pa, load_years, scaffold_folds, stratified_folds,
    stratified_split, time_split_indices,
)
from qsar_tutorial.featurizer import UniMolFeaturizer, featurize_ecfp4
from qsar_tutorial.model import build_classifier, cross_validated_oof, fit_with_class_weight
from qsar_tutorial.report import ReportPayload, render_report
from qsar_tutorial.sae import dead_ratio, descriptor_recovery, train_sae
from qsar_tutorial.shap_layer import explain


RDKIT_DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
    "HeavyAtomCount", "RingCount",
]


def compute_descriptors(smiles: np.ndarray) -> np.ndarray:
    out = np.zeros((len(smiles), len(RDKIT_DESCRIPTORS)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, name in enumerate(RDKIT_DESCRIPTORS):
            fn = getattr(Descriptors, name, None)
            try:
                v = float(fn(mol)) if fn else 0.0
            except Exception:
                v = 0.0
            out[i, j] = v if np.isfinite(v) else 0.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/coadd_pa.csv")
    p.add_argument("--features", choices=["ecfp4", "unimol"], default="ecfp4")
    p.add_argument("--out", default="reports/coadd_pa_report.html")
    p.add_argument("--max-n", type=int, default=None, help="Optional cap for fast iteration.")
    p.add_argument("--sae-latent", type=int, default=2048)
    p.add_argument("--sae-epochs", type=int, default=60)
    p.add_argument(
        "--split",
        choices=["auto", "stratified", "scaffold"],
        default="auto",
        help=(
            "Cross-validation split. 'auto' picks scaffold for SAR-dense data "
            "(activity-cliff ratio >= 0.05) and stratified otherwise. "
            "Force 'scaffold' for any single-target dataset (e.g. BRAF). "
            "See docs/EXPECTED_OUTPUTS.md '#data-regime' for rationale."
        ),
    )
    p.add_argument(
        "--time-cutoff", type=int, default=None,
        help=(
            "Additional time-based evaluation: train on year <= cutoff, "
            "eval on year > cutoff. Requires --year-csv (or processed CSV "
            "with min_document_year column)."
        ),
    )
    p.add_argument(
        "--year-csv", default=None,
        help=(
            "CSV providing min_document_year per molecule (default: auto-locate "
            "raw <name>_per_molecule.csv next to the processed CSV)."
        ),
    )
    p.add_argument(
        "--external", default=None,
        help="External CSV (canonical_smiles + active) for held-out evaluation.",
    )
    p.add_argument(
        "--metrics-json", default=None,
        help="Optional path to write a JSON dump of all metrics (for batch comparison).",
    )
    p.add_argument(
        "--cleanlab",
        action="store_true",
        help=(
            "Run active-protected Cleanlab cleaning on the current "
            "feature matrix before training. Off by default; pass this "
            "flag to flip flagged inactive labels to active. The active "
            "class is never relabeled (see src/qsar_tutorial/label_cleaning.py)."
        ),
    )
    args = p.parse_args()

    print(f"[0/6] Diagnostic — checking dataset suitability")
    try:
        import subprocess
        diag_result = subprocess.run(
            ["python", "scripts/diagnose_dataset.py", args.csv, "--json"],
            capture_output=True, text=True, timeout=120,
        )
        import json as _json
        diag = _json.loads(diag_result.stdout)
        cliff = diag.get("activity_cliffs", {}).get("fraction_on_cliff", 0.0)
        top10 = diag.get("scaffolds", {}).get("top10_share", 0.0)
        print(f"  cliff%={cliff:.1%}  top10_scaffold%={top10:.1%}")
        if cliff >= 0.10 and args.split == "stratified":
            print(f"  [WARN] activity_cliff_fraction {cliff:.1%} >= 10% with "
                  f"--split=stratified — leakage risk. Consider --split=scaffold.")
    except Exception as e:
        print(f"  [diagnose skip] {e}")

    print(f"[1/6] Loading {args.csv}")
    ds = load_coadd_pa(args.csv)
    if args.max_n:
        idx = np.random.default_rng(0).choice(len(ds), size=min(args.max_n, len(ds)), replace=False)
        ds = Dataset(smiles=ds.smiles[idx], y=ds.y[idx])
    print(f"  N={len(ds)}  active={ds.active_fraction:.1%}")

    print(f"[2/6] Featurizing ({args.features})")
    # Instantiate the Uni-Mol featurizer ONCE and reuse it everywhere
    # (featurize stage + MMP score_fn). Creating a new UniMolFeaturizer per
    # score_fn call reloads model weights + re-runs CPU conformer generation
    # on every MMP batch, leaving the GPU idle (see GH issue #1).
    unimol_featurizer = UniMolFeaturizer() if args.features == "unimol" else None
    if args.features == "ecfp4":
        X, valid = featurize_ecfp4(list(ds.smiles), n_bits=2048)
        feat_names = [f"ecfp_{i}" for i in range(X.shape[1])]
    else:
        X, valid = unimol_featurizer.featurize(list(ds.smiles))
        feat_names = [f"unimol_{i}" for i in range(X.shape[1])]
    X = X[valid]
    y = ds.y[valid]
    smi_valid = ds.smiles[valid]
    print(f"  X={X.shape}  y_active={y.mean():.1%}")

    if args.cleanlab:
        print("[2.5/6] Cleanlab (active-protected) on current features")
        from qsar_tutorial.label_cleaning import clean_labels_active_protected
        y_cleaned, cl_report = clean_labels_active_protected(X, y)
        print(
            f"  flagged={cl_report.n_flagged}, "
            f"relabeled 0->1={cl_report.n_relabeled}, "
            f"protected (active kept)={cl_report.n_protected}"
        )
        print(
            f"  active: {cl_report.active_before} -> "
            f"{cl_report.active_after} "
            f"({(cl_report.active_after - cl_report.active_before):+d})"
        )
        y = y_cleaned

    split_mode = args.split
    if split_mode == "auto":
        # Heuristic: the original CO-ADD path stays on stratified for
        # byte-compat reproducibility (low cliff fraction, large N).
        # Everything else defaults to scaffold split — the safe choice
        # for unknown / target-specific data.
        split_mode = "stratified" if "coadd" in args.csv.lower() else "scaffold"
    print(f"[3/6] Cross-validated OOF (5-fold, split={split_mode})")
    fold_ds = Dataset(smiles=smi_valid, y=y)
    if split_mode == "scaffold":
        folds = scaffold_folds(fold_ds, n_splits=5)
    else:
        folds = stratified_folds(fold_ds, n_splits=5)
    oof = cross_validated_oof(X, y, folds, n_estimators=200, max_depth=6)
    print(f"  {oof.summary()}")

    time_split_metrics = None
    if args.time_cutoff is not None:
        year_csv = args.year_csv
        if not year_csv:
            from pathlib import Path as _P
            stem = _P(args.csv).stem
            year_csv = f"data/raw/{stem}_per_molecule.csv"
        print(f"[3.5/6] Time-split eval (cutoff={args.time_cutoff}, year_csv={year_csv})")
        years = load_years(year_csv, smi_valid)
        tr_idx, te_idx = time_split_indices(years, args.time_cutoff)
        if len(te_idx) < 10 or len(tr_idx) < 10:
            print(f"  [warn] degenerate split (train={len(tr_idx)}, test={len(te_idx)}); skipping.")
        else:
            aux = build_classifier(n_estimators=200, max_depth=6)
            fit_with_class_weight(aux, X[tr_idx], y[tr_idx])
            p_te = aux.predict_proba(X[te_idx])[:, 1]
            time_split_metrics = dict(
                cutoff=int(args.time_cutoff),
                n_train=int(len(tr_idx)),
                n_test=int(len(te_idx)),
                train_active_pct=float(y[tr_idx].mean()),
                test_active_pct=float(y[te_idx].mean()),
                AUC=float(roc_auc_score(y[te_idx], p_te)),
                AUPRC=float(average_precision_score(y[te_idx], p_te)),
            )
            print(f"  train={len(tr_idx)} test={len(te_idx)} "
                  f"AUC={time_split_metrics['AUC']:.3f} "
                  f"AUPRC={time_split_metrics['AUPRC']:.3f}")

    print("[4/6] Fit final model + TreeSHAP")
    final = build_classifier(n_estimators=200, max_depth=6)
    fit_with_class_weight(final, X, y)
    shap_res = explain(final, X, feature_names=feat_names)
    top_shap = shap_res.top_features(k=20)

    shap_scaffold_diversity = None
    if args.features == "ecfp4":
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import Chem as _Chem
        scaffold_of = []
        for s in smi_valid:
            try:
                sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
                scaffold_of.append(sc if sc else f"__acyclic__{s}")
            except Exception:
                scaffold_of.append(f"__invalid__{s}")
        scaffold_arr = np.array(scaffold_of)
        shap_scaffold_diversity = []
        for name, _imp in top_shap[:10]:
            bit_idx = int(name.split("_")[-1])
            on_mask = X[:, bit_idx] > 0
            n_on = int(on_mask.sum())
            n_scaf = int(len(set(scaffold_arr[on_mask]))) if n_on else 0
            shap_scaffold_diversity.append({
                "feature": name,
                "n_compounds_on": n_on,
                "n_distinct_scaffolds": n_scaf,
                "scaffold_bound": n_scaf > 0 and n_scaf < 3,
            })

    print(f"[5/6] SAE (latent={args.sae_latent}, epochs={args.sae_epochs})")
    sae_fit = train_sae(
        X.astype(np.float32), latent_dim=args.sae_latent, epochs=args.sae_epochs, verbose=False
    )
    dr = dead_ratio(sae_fit, X.astype(np.float32))
    Z = sae_fit.encode(X.astype(np.float32))
    descs = compute_descriptors(smi_valid)
    r2 = descriptor_recovery(Z, descs)
    sae_summary = dict(
        latent_dim=args.sae_latent,
        dead_ratio=float(dr),
        r2_median=float(np.median(r2)),
        r2_above_05=int(np.sum(r2 > 0.5)),
        r2_total=int(len(r2)),
    )
    print(f"  dead_ratio={dr:.3f}  R²_median={sae_summary['r2_median']:.3f}")

    print("[6/6] MMP counterfactual + HTML report")
    inactive = list(smi_valid[y == 0][: min(500, int((y == 0).sum()))])

    def score_fn(smis):
        if args.features == "ecfp4":
            Xs, vs = featurize_ecfp4(smis, n_bits=X.shape[1])
            proba = final.predict_proba(Xs)[:, 1]
            proba[~vs] = 0.0
            return proba
        Xs, vs = unimol_featurizer.featurize(smis)
        proba = np.zeros(len(smis))
        if vs.any():
            proba[vs] = final.predict_proba(Xs[vs])[:, 1]
        return proba

    mmp = scan(inactive, score_fn)
    mmp_rows = [
        dict(
            name=r.name, smirks=r.smirks, n_applied=r.n_applied,
            delta_p_mean=r.delta_p_mean, delta_p_ci95=r.delta_p_ci95,
            exemplar_delta=r.exemplar_delta,
            n_distinct_scaffolds=r.n_distinct_scaffolds,
            is_series_local=r.is_series_local,
        )
        for r in mmp
    ]

    external_metrics = None
    if args.external:
        print(f"[6.5/6] External hold-out eval on {args.external}")
        ext_ds = load_coadd_pa(args.external)
        if args.features == "ecfp4":
            X_ext, v_ext = featurize_ecfp4(list(ext_ds.smiles), n_bits=X.shape[1])
        else:
            X_ext, v_ext = UniMolFeaturizer().featurize(list(ext_ds.smiles))
        X_ext = X_ext[v_ext]
        y_ext = ext_ds.y[v_ext]
        if len(y_ext) == 0 or len(np.unique(y_ext)) < 2:
            print(f"  [warn] external set degenerate (n={len(y_ext)}); skipping.")
        else:
            p_ext = final.predict_proba(X_ext)[:, 1]
            ext_auc = float(roc_auc_score(y_ext, p_ext))
            oof_auc = float(oof.summary()["AUC"])
            gap = oof_auc - ext_auc
            if gap >= 0.20:
                gap_level = "severe"
            elif gap >= 0.10:
                gap_level = "moderate"
            else:
                gap_level = "ok"
            external_metrics = dict(
                csv=args.external,
                n=int(len(y_ext)),
                active_pct=float(y_ext.mean()),
                AUC=ext_auc,
                AUPRC=float(average_precision_score(y_ext, p_ext)),
                oof_auc=oof_auc,
                auc_gap=gap,
                gap_level=gap_level,
            )
            print(f"  n={len(y_ext)} active={y_ext.mean():.1%} "
                  f"AUC={external_metrics['AUC']:.3f} "
                  f"AUPRC={external_metrics['AUPRC']:.3f}")

    title_suffix = (
        f" — {args.features.upper()} pipeline (split={split_mode})"
    )
    payload = ReportPayload(
        title=f"{Path(args.csv).stem}{title_suffix}",
        dataset_name=args.csv,
        n_total=len(y),
        metrics=oof.summary(),
        top_shap=top_shap,
        sae=sae_summary,
        mmp_rows=mmp_rows,
        time_split=time_split_metrics,
        external=external_metrics,
        shap_scaffold_diversity=shap_scaffold_diversity,
    )
    out_path = render_report(payload, args.out)
    print(f"\nReport written: {out_path}")

    if args.metrics_json:
        import json
        with open(args.metrics_json, "w") as f:
            json.dump(
                {
                    "csv": args.csv,
                    "features": args.features,
                    "split_mode": split_mode,
                    "n": int(len(y)),
                    "active_pct": float(y.mean()),
                    "oof": oof.summary(),
                    "sae": sae_summary,
                    "mmp_n_rules_excl_zero": int(
                        sum(
                            1 for r in mmp_rows
                            if r["delta_p_ci95"][0] > 0 or r["delta_p_ci95"][1] < 0
                        )
                    ),
                    "mmp_n_cross_series": int(
                        sum(1 for r in mmp_rows if not r.get("is_series_local"))
                    ),
                    "mmp_n_series_local": int(
                        sum(1 for r in mmp_rows if r.get("is_series_local"))
                    ),
                    "time_split": time_split_metrics,
                    "external": external_metrics,
                },
                f, indent=2,
            )
        print(f"Metrics JSON: {args.metrics_json}")


if __name__ == "__main__":
    main()
