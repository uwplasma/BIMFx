from __future__ import annotations

from typing import Any

import numpy as np


def field_alignment_error(
    psi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    B: Any,
) -> np.ndarray:
    """Compute q = abs(t dot grad(psi)) / abs(grad(psi)) on the grid."""
    dpsi_dx, dpsi_dy, dpsi_dz = np.gradient(psi, xs, ys, zs, edge_order=1)
    grad = np.stack([dpsi_dx, dpsi_dy, dpsi_dz], axis=-1)
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    Bv = np.asarray(B(pts))
    if Bv.shape == (3,):
        Bv = Bv[None, :]
    Bn = Bv / np.maximum(1e-30, np.linalg.norm(Bv, axis=1, keepdims=True))
    t = Bn.reshape(grad.shape)

    dot = np.abs(np.sum(t * grad, axis=-1))
    gnorm = np.linalg.norm(grad, axis=-1)
    return dot / np.maximum(1e-30, gnorm)
