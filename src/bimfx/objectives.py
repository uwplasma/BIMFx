from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

Array = Any

__all__ = ["boundary_residual_objective", "divergence_objective"]

def boundary_residual_objective(
    B: Callable[[Array], Array],
    P: Array,
    N: Array,
    *,
    eps_factor: float = 0.02,
) -> Array:
    """Mean-squared boundary residual using inward-offset points."""
    P = jnp.asarray(P)
    N = jnp.asarray(N)
    center = jnp.mean(P, axis=0)
    scale = jnp.median(jnp.linalg.norm(P - center, axis=1))
    Pin = P - eps_factor * scale * N
    Bv = jnp.asarray(B(Pin))
    if Bv.shape == (3,):
        Bv = Bv[None, :]
    ndot = jnp.sum(N * Bv, axis=1)
    denom = jnp.median(jnp.linalg.norm(Bv, axis=1))
    res = ndot / jnp.maximum(1e-30, denom)
    return jnp.mean(res**2)


def divergence_objective(
    B: Callable[[Array], Array],
    xs: Array,
    ys: Array,
    zs: Array,
) -> Array:
    """Mean-squared divergence on a Cartesian grid."""
    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
    pts = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    Bv = jnp.asarray(B(pts))
    if Bv.shape == (3,):
        Bv = Bv[None, :]
    Bv = Bv.reshape(X.shape + (3,))
    dBx_dx = jnp.gradient(Bv[..., 0], xs, axis=0, edge_order=1)
    dBy_dy = jnp.gradient(Bv[..., 1], ys, axis=1, edge_order=1)
    dBz_dz = jnp.gradient(Bv[..., 2], zs, axis=2, edge_order=1)
    divB = dBx_dx + dBy_dy + dBz_dz
    return jnp.mean(divB**2)
