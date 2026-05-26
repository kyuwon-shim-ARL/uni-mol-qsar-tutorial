"""Smoke tests: imports + counterfactual SMIRKS run without GPU/heavy deps.

These tests intentionally avoid Uni-Mol and SAE training so CI can run
without unimol-tools or a GPU. The heavier paths are exercised through
the example script and notebooks.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_package_imports():
    import qsar_tutorial
    from qsar_tutorial import data, featurizer, model, shap_layer, sae, counterfactual, report  # noqa
    assert qsar_tutorial.__version__


def test_ecfp4_featurizer_shape():
    from qsar_tutorial.featurizer import featurize_ecfp4

    smis = ["CCO", "c1ccccc1", "invalid_smiles", "CCN(CC)CC"]
    X, valid = featurize_ecfp4(smis, n_bits=128)
    assert X.shape == (4, 128)
    assert valid.tolist() == [True, True, False, True]
    assert X[2].sum() == 0  # invalid row zero-filled


def test_counterfactual_apply_transform():
    from qsar_tutorial.counterfactual import apply_transform

    # add COOH to benzene → benzoic acid
    products = apply_transform("c1ccccc1", "[c;H1:1]>>[c:1]C(=O)O")
    assert len(products) >= 1
    assert "C(=O)O" in products[0] or "O=C(O)" in products[0]


def test_counterfactual_scan_with_dummy_scorer():
    from qsar_tutorial.counterfactual import scan, DEFAULT_TRANSFORMATIONS

    def dummy_score(smis):
        # toy scorer: more 'O' chars → higher activity
        return np.array([min(s.count("O") / 5.0, 1.0) for s in smis])

    parents = ["c1ccccc1", "CCN", "CCO", "Nc1ccccc1"]
    out = scan(parents, dummy_score, transformations={
        "add_COOH_aromatic": DEFAULT_TRANSFORMATIONS["add_COOH_aromatic"],
    })
    assert len(out) == 1
    r = out[0]
    assert r.n_applied >= 1
    # adding COOH to aromatic carbons should raise the score under our dummy
    assert r.delta_p_mean > 0


def test_model_build():
    pytest.importorskip("xgboost")
    from qsar_tutorial.model import build_classifier

    clf = build_classifier(n_estimators=10, max_depth=3)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
