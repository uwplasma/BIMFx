from __future__ import annotations

from typing import Any, Callable

import numpy as np


def offset_points_inward(
    P: np.ndarray,
    N: np.ndarray,
    *,
    eps_factor: float = 0.02,
) -> np.ndarray:
    """Offset boundary points inward by eps_factor * median radius."""
    P = np.asarray(P, dtype=float)
    N = np.asarray(N, dtype=float)
    center = np.mean(P, axis=0)
    scale = np.median(np.linalg.norm(P - center[None, :], axis=1))
    return P - eps_factor * scale * N


def boundary_normal_residual(
    B: Callable[[Any], Any],
    P: np.ndarray,
    N: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Compute |n·B| (optionally normalized by |B|) on boundary points."""
    P = np.asarray(P, dtype=float)
    N = np.asarray(N, dtype=float)
    Bv = np.asarray(B(P))
    if Bv.shape == (3,):
        Bv = Bv[None, :]
    ndot = np.sum(N * Bv, axis=1)
    if not normalize:
        return np.abs(ndot)
    denom = np.linalg.norm(Bv, axis=1)
    return np.abs(ndot) / np.maximum(1e-30, denom)


def relative_boundary_residual(
    B: Callable[[Any], Any],
    P: np.ndarray,
    N: np.ndarray,
    *,
    eps_factor: float = 0.02,
) -> np.ndarray:
    """Compute |n·B| normalized by median |B| on inward-offset points."""
    Pin = offset_points_inward(P, N, eps_factor=eps_factor)
    Bv = np.asarray(B(Pin))
    if Bv.shape == (3,):
        Bv = Bv[None, :]
    ndot = np.sum(N * Bv, axis=1)
    scale = np.median(np.linalg.norm(Bv, axis=1))
    return np.abs(ndot) / max(scale, 1e-30)


def divergence_on_grid(
    B: Callable[[Any], Any],
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> np.ndarray:
    """Compute div(B) on a Cartesian grid using finite differences."""
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    Bv = np.asarray(B(pts))
    if Bv.shape == (3,):
        Bv = Bv[None, :]
    Bv = Bv.reshape(X.shape + (3,))

    dBx_dx = np.gradient(Bv[..., 0], xs, axis=0, edge_order=1)
    dBy_dy = np.gradient(Bv[..., 1], ys, axis=1, edge_order=1)
    dBz_dz = np.gradient(Bv[..., 2], zs, axis=2, edge_order=1)
    return dBx_dx + dBy_dy + dBz_dz


def summary_stats(values: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for a scalar field."""
    vals = np.asarray(values).ravel()
    return {
        "min": float(np.min(vals)),
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "p95": float(np.percentile(vals, 95.0)),
        "max": float(np.max(vals)),
        "rms": float(np.sqrt(np.mean(vals**2))),
    }


def validate_vacuum_field(
    B: Callable[[Any], Any],
    P: np.ndarray,
    N: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Return summary stats for boundary residual and div(B) on a grid."""
    bres = boundary_normal_residual(B, P, N, normalize=True)
    divB = divergence_on_grid(B, xs, ys, zs)
    if mask is not None:
        div_vals = divB[mask]
    else:
        div_vals = divB
    return {
        "boundary_normal_residual": summary_stats(bres),
        "divergence": summary_stats(div_vals),
    }
