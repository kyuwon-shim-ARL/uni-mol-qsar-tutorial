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

from qsar_tutorial.counterfactual import scan
from qsar_tutorial.data import Dataset, load_coadd_pa, stratified_folds, stratified_split
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

    print(f"[1/6] Loading {args.csv}")
    ds = load_coadd_pa(args.csv)
    if args.max_n:
        idx = np.random.default_rng(0).choice(len(ds), size=min(args.max_n, len(ds)), replace=False)
        ds = Dataset(smiles=ds.smiles[idx], y=ds.y[idx])
    print(f"  N={len(ds)}  active={ds.active_fraction:.1%}")

    print(f"[2/6] Featurizing ({args.features})")
    if args.features == "ecfp4":
        X, valid = featurize_ecfp4(list(ds.smiles), n_bits=2048)
        feat_names = [f"ecfp_{i}" for i in range(X.shape[1])]
    else:
        feat = UniMolFeaturizer()
        X, valid = feat.featurize(list(ds.smiles))
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

    print("[3/6] Cross-validated OOF (5-fold)")
    folds = stratified_folds(Dataset(smiles=smi_valid, y=y), n_splits=5)
    oof = cross_validated_oof(X, y, folds, n_estimators=200, max_depth=6)
    print(f"  {oof.summary()}")

    print("[4/6] Fit final model + TreeSHAP")
    final = build_classifier(n_estimators=200, max_depth=6)
    fit_with_class_weight(final, X, y)
    shap_res = explain(final, X, feature_names=feat_names)
    top_shap = shap_res.top_features(k=20)

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
        feat = UniMolFeaturizer()
        Xs, vs = feat.featurize(smis)
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
        )
        for r in mmp
    ]

    payload = ReportPayload(
        title=f"CO-ADD PA — {args.features.upper()} pipeline",
        dataset_name="CO-ADD P. aeruginosa (public)",
        n_total=len(y),
        metrics=oof.summary(),
        top_shap=top_shap,
        sae=sae_summary,
        mmp_rows=mmp_rows,
    )
    out_path = render_report(payload, args.out)
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
