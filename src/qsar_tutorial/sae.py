"""Sparse Autoencoder for decomposing dense foundation-model embeddings.

Design choices (lessons learned the hard way in the parent project):

  * Z-score normalize inputs *before* training. Without this, ~60% of latents
    die during training.
  * Mini-batch via DataLoader (batch_size=256). Full-batch training also
    starves latents.
  * L1 sparsity penalty on the post-ReLU code. Tune lambda so dead_ratio
    settles below 0.05.
  * 200 epochs is usually enough on 5k-50k molecules.

After training, evaluate features by Ridge-regressing each RDKit physicochem
descriptor on the SAE code. Median R² should be > 0.5 for a usable SAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 6144, tied: bool = True):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.bias_dec = nn.Parameter(torch.zeros(input_dim))
        self.tied = tied
        if not tied:
            self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.tied:
            return torch.nn.functional.linear(z, self.encoder.weight.T) + self.bias_dec
        return self.decoder(z) + self.bias_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


@dataclass
class SAEFit:
    model: SparseAutoencoder
    mean: np.ndarray
    std: np.ndarray
    losses: list[float]

    def encode(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mean) / (self.std + 1e-8)
        with torch.no_grad():
            z = self.model.encode(torch.as_tensor(Xn, dtype=torch.float32))
        return z.cpu().numpy()


def train_sae(
    X: np.ndarray,
    latent_dim: int = 6144,
    l1_lambda: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | None = None,
    verbose: bool = True,
) -> SAEFit:
    """Z-score + mini-batch SAE training. Returns SAEFit with stats for inference."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Xn = (X - mean) / (std + 1e-8)

    ds = TensorDataset(torch.as_tensor(Xn, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SparseAutoencoder(input_dim=X.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        ep_loss = 0.0
        n_batches = 0
        for (xb,) in dl:
            xb = xb.to(device)
            x_hat, z = model(xb)
            recon = torch.mean((x_hat - xb) ** 2)
            sparsity = z.abs().mean()
            loss = recon + l1_lambda * sparsity
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        losses.append(ep_loss / max(n_batches, 1))
        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(f"epoch {epoch:4d}  loss={losses[-1]:.4f}")

    return SAEFit(model=model.cpu(), mean=mean, std=std, losses=losses)


def dead_ratio(fit: SAEFit, X: np.ndarray, threshold: float = 1e-6) -> float:
    """Fraction of latents that activate (>threshold) on no sample."""
    Z = fit.encode(X)
    alive = (Z.max(axis=0) > threshold)
    return float(1.0 - alive.mean())


def descriptor_recovery(
    Z: np.ndarray, descriptors: np.ndarray, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    """5-fold CV Ridge R² for each descriptor column. Returns array of R² per descriptor."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s = np.zeros(descriptors.shape[1])
    for j in range(descriptors.shape[1]):
        y = descriptors[:, j]
        preds = np.zeros_like(y)
        for tr, va in kf.split(Z):
            r = Ridge(alpha=1.0).fit(Z[tr], y[tr])
            preds[va] = r.predict(Z[va])
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2s[j] = 1.0 - ss_res / (ss_tot + 1e-12)
    return r2s
