"""Featurizers: Uni-Mol 3D embedding (main) + ECFP4 baseline.

The educational point: the *same* downstream pipeline (XGBoost + SHAP) works
on either feature set. Compare them side-by-side to see the cost/benefit of
the 3D foundation model versus the classical fingerprint.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


def featurize_ecfp4(smiles: List[str], n_bits: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """ECFP4 baseline. Returns (X, valid_mask).

    valid_mask is True for SMILES that parsed; rows in X are aligned with
    the input list (invalid rows are zero-filled and masked out).
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    X = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        X[i] = np.frombuffer(bytes(fp.ToBitString(), "utf-8"), dtype=np.uint8) - ord("0")
        valid[i] = True
    return X, valid


class UniMolFeaturizer:
    """Wrap unimol-tools UniMolRepr for 512-d (v1) or 768-d (v2) embeddings.

    The class is constructed lazily so this module imports without requiring
    `unimol-tools` to be installed (useful for the ECFP-only path).
    """

    def __init__(
        self,
        model_name: str = "unimolv1",
        model_size: str = "84m",
        batch_size: int = 32,
        use_gpu: bool | None = None,
    ):
        self.model_name = model_name
        self.model_size = model_size
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self._model = None

    def _load(self) -> None:
        from unimol_tools import UniMolRepr

        if self.use_gpu is None:
            import torch

            self.use_gpu = torch.cuda.is_available()
        self._model = UniMolRepr(
            data_type="molecule",
            remove_hs=False,
            model_name=self.model_name,
            model_size=self.model_size,
            use_gpu=self.use_gpu,
        )

    def featurize(self, smiles: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, valid_mask). X.shape = (N, D) with D=512 (v1) or 768 (v2)."""
        if self._model is None:
            self._load()

        n = len(smiles)
        embs: list[np.ndarray | None] = [None] * n
        valid = np.zeros(n, dtype=bool)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = smiles[start:end]
            try:
                reps = self._model.get_repr(batch, return_atomic_reprs=False)
            except Exception:
                reps = [None] * len(batch)
            for j, emb in enumerate(reps):
                if emb is None:
                    continue
                arr = np.asarray(emb)
                if np.isnan(arr).any():
                    continue
                embs[start + j] = arr
                valid[start + j] = True

        # Stack with zero-fill for invalid rows so indices stay aligned.
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            raise RuntimeError("All Uni-Mol embeddings failed.")
        dim = embs[valid_idx[0]].shape[-1]
        X = np.zeros((n, dim), dtype=np.float32)
        for i in valid_idx:
            X[i] = embs[i]
        return X, valid
