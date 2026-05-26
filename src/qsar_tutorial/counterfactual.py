"""Matched Molecular Pair (MMP) design-rule scanning — binary version.

For a binary active/inactive task we ask one direction only:
"which substitution turns *inactive* compounds into predicted *active* ones?"

Pipeline:
  1. Enumerate single-substitution transformations from a curated SMIRKS list.
  2. Apply each to every inactive parent.
  3. Score parent vs child with the trained model; record Δp(active).
  4. Aggregate per-transformation: count, mean Δp, BCa 95% CI.

Effect-size CIs use a simple bootstrap so the rule of thumb "if 95% CI
excludes 0, treat as a hypothesis, NOT a validated rule" is enforceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


# Educational starter set. Easy to extend.
DEFAULT_TRANSFORMATIONS: dict[str, str] = {
    "add_COOH_aromatic":      "[c;H1:1]>>[c:1]C(=O)O",
    "add_F_aromatic":          "[c;H1:1]>>[c:1]F",
    "add_Cl_aromatic":         "[c;H1:1]>>[c:1]Cl",
    "add_OH_aromatic":         "[c;H1:1]>>[c:1]O",
    "add_methyl_aromatic":     "[c;H1:1]>>[c:1]C",
    "add_NH2_aromatic":        "[c;H1:1]>>[c:1]N",
    "amidine_to_propargylamine":
        "[NX3:1][CX3:2]=[NX2:3]>>[NX3:1]CC#C",
    "primary_amine_to_propargyl":
        "[NX3;H2:1][CX4:2]>>[NX3:1](CC#C)[CX4:2]",
}


@dataclass
class TransformResult:
    name: str
    smirks: str
    n_applied: int
    n_scored: int
    delta_p_mean: float
    delta_p_ci95: tuple[float, float]
    exemplar_smiles: str | None
    exemplar_delta: float


def apply_transform(parent_smiles: str, smirks: str, max_products: int = 1) -> list[str]:
    """Apply a SMIRKS to a parent SMILES; return canonical product SMILES."""
    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        return []
    rxn = AllChem.ReactionFromSmarts(smirks)
    if rxn is None:
        return []
    products: list[str] = []
    try:
        outcomes = rxn.RunReactants((parent,))
    except Exception:
        return []
    seen: set[str] = set()
    for tuple_ in outcomes:
        for mol in tuple_:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            smi = Chem.MolToSmiles(mol, canonical=True)
            if smi and smi not in seen:
                seen.add(smi)
                products.append(smi)
                if len(products) >= max_products:
                    return products
    return products


def _bca_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    """Simple percentile bootstrap CI. (BCa proper omitted for tutorial clarity.)"""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = float(np.mean(values[idx]))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def scan(
    inactive_smiles: Iterable[str],
    score_fn: Callable[[list[str]], np.ndarray],
    transformations: dict[str, str] | None = None,
) -> list[TransformResult]:
    """For each transformation, apply to all inactive parents and aggregate Δp(active).

    Args:
        inactive_smiles : SMILES of inactive parents to perturb.
        score_fn        : callable that takes a list of SMILES and returns
                          an array of P(active) of the same length.
        transformations : dict[name → SMIRKS]. Defaults to DEFAULT_TRANSFORMATIONS.

    Returns:
        List of TransformResult, one per transformation.
    """
    transformations = transformations or DEFAULT_TRANSFORMATIONS
    parents = list(inactive_smiles)
    parent_scores = score_fn(parents)
    parent_score_by_smi = dict(zip(parents, parent_scores))

    results: list[TransformResult] = []
    for name, smirks in transformations.items():
        deltas: list[float] = []
        n_applied = 0
        best_smi = None
        best_delta = -np.inf

        # First pass: enumerate all products.
        all_children: list[str] = []
        owner: list[tuple[str, int]] = []  # (parent_smi, idx into deltas after scoring)
        for p_smi in parents:
            products = apply_transform(p_smi, smirks, max_products=1)
            for c_smi in products:
                if c_smi == p_smi:
                    continue
                all_children.append(c_smi)
                owner.append((p_smi, len(all_children) - 1))
                n_applied += 1

        if n_applied == 0:
            results.append(TransformResult(name, smirks, 0, 0, 0.0, (float("nan"),) * 2, None, 0.0))
            continue

        # Single batched score call for efficiency.
        child_scores = score_fn(all_children)

        for (p_smi, k), c_smi, c_score in zip(owner, all_children, child_scores):
            d = float(c_score - parent_score_by_smi[p_smi])
            deltas.append(d)
            if d > best_delta:
                best_delta = d
                best_smi = c_smi

        deltas_arr = np.asarray(deltas)
        lo, hi = _bca_ci(deltas_arr)
        results.append(
            TransformResult(
                name=name,
                smirks=smirks,
                n_applied=n_applied,
                n_scored=len(deltas_arr),
                delta_p_mean=float(deltas_arr.mean()),
                delta_p_ci95=(lo, hi),
                exemplar_smiles=best_smi,
                exemplar_delta=float(best_delta),
            )
        )
    return results
