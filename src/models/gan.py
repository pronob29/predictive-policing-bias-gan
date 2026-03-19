"""
src/models/gan.py
=================
GAN architecture for geographic crime location modelling.

Architecture (matching paper):
  Generator:      noise(100) → FC(256) → BN → LeakyReLU
                            → FC(512) → BN → LeakyReLU
                            → FC(256) → BN → LeakyReLU
                            → FC(2)   → Tanh
  Discriminator:  (lat,lon) → FC(512) → LeakyReLU → Dropout(0.3)
                            → FC(256) → LeakyReLU → Dropout(0.3)
                            → FC(128) → LeakyReLU
                            → FC(1)   → Sigmoid
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple


# ── Model Definitions ─────────────────────────────────────────────────────────

class Generator(nn.Module):
    """Maps latent noise to 2-D geographic coordinates (lat, lon)."""

    def __init__(self, noise_dim: int = 100, output_dim: int = 2,
                 hidden_dims: Tuple[int, ...] = (256, 512, 256)):
        super().__init__()
        layers = []
        in_dim = noise_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2)]
            in_dim = h
        layers += [nn.Linear(in_dim, output_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Classifies real vs generated geographic coordinates."""

    def __init__(self, input_dim: int = 2,
                 hidden_dims: Tuple[int, ...] = (512, 256, 128),
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i, h in enumerate(hidden_dims):
            layers += [nn.Linear(in_dim, h), nn.LeakyReLU(0.2)]
            if i < 2:                        # dropout on first two blocks only
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Coordinate Normalisation ──────────────────────────────────────────────────

def normalise_coords(
    coords: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise lat/lon coordinates for GAN training."""
    if mean is None:
        mean = coords.mean(axis=0)
    if std is None:
        std = coords.std(axis=0)
        std = np.where(std == 0, 1.0, std)
    return (coords - mean) / std, mean, std


def denormalise_coords(norm_coords: np.ndarray,
                       mean: np.ndarray,
                       std: np.ndarray) -> np.ndarray:
    """Recover original lat/lon from normalised coordinates."""
    return norm_coords * std + mean


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_gan(
    real_coords: np.ndarray,
    noise_dim: int = 100,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 0.0002,
    betas: Tuple[float, float] = (0.5, 0.999),
    device: Optional[torch.device] = None,
) -> Tuple[Optional[Generator], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Train GAN on real crime location coordinates.

    Parameters
    ----------
    real_coords : ndarray, shape (N, 2)  — (latitude, longitude) pairs
    noise_dim   : latent dimension
    epochs      : training iterations
    batch_size  : mini-batch size (clipped to N if N < batch_size)
    lr          : Adam learning rate
    betas       : Adam momentum parameters
    device      : torch device (defaults to CPU)

    Returns
    -------
    generator   : trained Generator (or None if too few samples)
    mean        : coordinate mean used for normalisation
    std         : coordinate std  used for normalisation
    """
    if device is None:
        device = torch.device("cpu")

    n = len(real_coords)
    if n < 10:
        return None, None, None

    norm_coords, mean, std = normalise_coords(real_coords)
    real_tensor = torch.FloatTensor(norm_coords).to(device)

    G = Generator(noise_dim=noise_dim).to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=betas)
    criterion = nn.BCELoss()

    bs = min(batch_size, n)

    for _ in range(epochs):
        idx = torch.randint(0, n, (bs,))
        real_b = real_tensor[idx]
        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        # Discriminator step
        z = torch.randn(bs, noise_dim, device=device)
        fake_b = G(z).detach()
        D.zero_grad()
        loss_D = criterion(D(real_b), real_labels) + criterion(D(fake_b), fake_labels)
        loss_D.backward()
        opt_D.step()

        # Generator step
        G.zero_grad()
        z = torch.randn(bs, noise_dim, device=device)
        loss_G = criterion(D(G(z)), real_labels)
        loss_G.backward()
        opt_G.step()

    return G, mean, std


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_patrol_locations(
    generator: Generator,
    mean: np.ndarray,
    std: np.ndarray,
    n_officers: int = 60,
    noise_dim: int = 100,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Sample n_officers patrol locations from the trained Generator.

    Returns
    -------
    locations : ndarray, shape (n_officers, 2) — (latitude, longitude)
    """
    if device is None:
        device = torch.device("cpu")
    with torch.no_grad():
        z = torch.randn(n_officers, noise_dim, device=device)
        fake_norm = generator(z).cpu().numpy()
    return denormalise_coords(fake_norm, mean, std)
