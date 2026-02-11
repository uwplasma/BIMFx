from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MFSKernelCache:
    """Precompute Green's function matrices for fast repeated evaluations."""

    G: np.ndarray           # (N,M)
    gradG: np.ndarray       # (N,M,3)

    @classmethod
    def from_points(cls, X: np.ndarray, Y: np.ndarray) -> "MFSKernelCache":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        diff = X[:, None, :] - Y[None, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        r = np.sqrt(np.maximum(r2, 1e-30))
        G = 1.0 / (4.0 * np.pi * r)
        r3 = np.maximum(1e-30, r2 * r)
        gradG = -diff / (4.0 * np.pi * r3[..., None])
        return cls(G=G, gradG=gradG)

    def phi(self, alpha: np.ndarray) -> np.ndarray:
        return self.G @ np.asarray(alpha)

    def grad(self, alpha: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha)
        return np.sum(self.gradG * alpha[None, :, None], axis=1)
