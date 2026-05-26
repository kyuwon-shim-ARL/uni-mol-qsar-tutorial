"""qsar_tutorial — Interpretable QSAR on CO-ADD PA binary data.

Modules:
    data        Load + split CO-ADD P. aeruginosa public data.
    featurizer  Uni-Mol embedding and ECFP4 baseline featurizers.
    model       XGBoost binary classifier (TreeSHAP-compatible).
    shap_layer  TreeSHAP wrapper + per-feature attributions.
    sae         Sparse Autoencoder for embedding decomposition.
    counterfactual  Matched Molecular Pair design-rule scanning.
    report      Self-contained HTML report generator.
"""

__version__ = "0.1.0"
