"""Load and split the CO-ADD P. aeruginosa public dataset.

CO-ADD (Community for Open Antimicrobial Drug Discovery) publishes phenotypic
screens against P. aeruginosa under an academic-research-friendly license.
We expect a CSV with at minimum:

    canonical_smiles  : SMILES string
    active            : 0 / 1  binary activity label

See scripts/prepare_data.py for how to obtain and normalize the source files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class Dataset:
    smiles: np.ndarray  # shape (N,)
    y: np.ndarray       # shape (N,), {0, 1}

    def __len__(self) -> int:
        return len(self.smiles)

    @property
    def active_fraction(self) -> float:
        return float(np.mean(self.y))


def load_coadd_pa(csv_path: str | Path) -> Dataset:
    """Load the CO-ADD PA combined per-molecule CSV.

    Drops rows with invalid SMILES or missing labels. Canonicalizes SMILES
    via RDKit so duplicates collapse.
    """
    df = pd.read_csv(csv_path)
    required = {"canonical_smiles", "active"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=["canonical_smiles", "active"]).copy()
    df["active"] = df["active"].astype(int)

    canon = []
    keep = []
    for smi in df["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            canon.append(None)
            keep.append(False)
            continue
        canon.append(Chem.MolToSmiles(mol, canonical=True))
        keep.append(True)
    df["canonical_smiles"] = canon
    df = df.loc[keep].drop_duplicates(subset="canonical_smiles").reset_index(drop=True)

    return Dataset(
        smiles=df["canonical_smiles"].to_numpy(),
        y=df["active"].to_numpy().astype(int),
    )


def stratified_split(
    ds: Dataset, test_size: float = 0.2, seed: int = 42
) -> tuple[Dataset, Dataset]:
    """Stratified train/test split preserving the active/inactive ratio."""
    train_idx, test_idx = train_test_split(
        np.arange(len(ds)),
        test_size=test_size,
        stratify=ds.y,
        random_state=seed,
    )
    return (
        Dataset(smiles=ds.smiles[train_idx], y=ds.y[train_idx]),
        Dataset(smiles=ds.smiles[test_idx], y=ds.y[test_idx]),
    )


def stratified_folds(
    ds: Dataset, n_splits: int = 5, seed: int = 42
) -> Sequence[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) tuples for stratified K-fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(ds.y), ds.y))


def scaffold_folds(
    ds: Dataset, n_splits: int = 5, seed: int = 42
) -> Sequence[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold over Bemis-Murcko scaffolds — molecules with the same
    scaffold cannot appear in both train and validation of a fold.

    Why: random K-fold on chemistry data overestimates AUC because close
    analogues leak between folds. Scaffold split is the standard defence.
    Expect AUC to drop 0.05-0.15 vs the random split on this dataset.

    Notes:
    - GroupKFold itself is deterministic by group order. We shuffle the
      group ordering with `seed` so different seeds give different splits.
    - Fold-level active counts are NOT guaranteed equal (scaffold groups
      are not stratified) — this is intentional. The per-fold active count
      is logged so the caller can warn on degenerate folds.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from sklearn.model_selection import GroupKFold

    group_keys: list[str] = []
    for smi in ds.smiles:
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smi, includeChirality=False
            )
        except Exception:
            sc = ""
        # Acyclic molecules produce an empty scaffold — fall back to the
        # SMILES itself so each acyclic molecule is its own group (no
        # accidental merging of unrelated molecules into one bucket).
        group_keys.append(sc if sc else f"acyclic::{smi}")

    unique = list(dict.fromkeys(group_keys))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    group_to_int = {g: i for i, g in enumerate(unique)}
    groups = np.array([group_to_int[g] for g in group_keys])

    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros_like(ds.y), ds.y, groups=groups))


def fold_label_counts(ds: Dataset, folds) -> list[dict]:
    """Per-fold (n_train, n_val, n_active_*) — surface degenerate folds.

    Scaffold splits can produce folds with zero positives in val; that
    fold's AUC is undefined. Log this so the report can footnote it.
    """
    rows = []
    for k, (tr, va) in enumerate(folds, start=1):
        rows.append({
            "fold": k,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "n_active_train": int(ds.y[tr].sum()),
            "n_active_val": int(ds.y[va].sum()),
        })
    return rows
