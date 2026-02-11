from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bimfx.mfs.geometry import (
    detect_geometry_and_axis,
    kNN_geometry_stats,
    maybe_flip_normals,
    multivalued_bases_about_axis,
    normalize_geometry,
)
from bimfx.mfs.sources_kernels import build_evaluators_mfs, build_evaluators_mfs_accel, build_mfs_sources
from bimfx.mfs.solvers import (
    build_system_matrices,
    fit_mv_coeffs_minimize_rhs,
    solve_alpha_with_rhs,
)


@dataclass(frozen=True)
class MFSSolution:
    phi: Any
    B: Any
    metadata: dict[str, Any]


def solve_mfs_neumann(
    points: Any,
    normals: Any,
    *,
    use_multivalued: bool = True,
    k_nn: int = 48,
    source_factor: float = 2.0,
    lambda_reg: float = 1e-6,
    harmonic_coeffs: tuple[float, float] | None = None,
    acceleration: str = "none",
    accel_theta: float = 0.6,
    accel_leaf_size: int = 64,
    verbose: bool = True,
) -> MFSSolution:
    """MFS-based Neumann solve enforcing ``n·B = 0`` on the boundary point cloud.

    Set ``acceleration="barnes-hut"`` to enable approximate far-field summation.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "BIMFx MFS solver requires JAX 64-bit mode for numerical stability. "
            "Set `JAX_ENABLE_X64=1` in your environment (recommended) or call "
            "`jax.config.update('jax_enable_x64', True)` before importing `bimfx`."
        )
    P = jnp.asarray(points, dtype=jnp.float64)
    N = jnp.asarray(normals, dtype=jnp.float64)
    N = N / jnp.maximum(1e-30, jnp.linalg.norm(N, axis=1, keepdims=True))

    N, flipped = maybe_flip_normals(P, N)
    Pn, scinfo = normalize_geometry(P, verbose=verbose)
    Nn = N

    W, rk = kNN_geometry_stats(Pn, k=k_nn, verbose=verbose)
    Yn, delta_n = build_mfs_sources(Pn, Nn, rk, scinfo, source_factor=source_factor, verbose=verbose)

    kind, a_hat, _E_axes, _c_axes, svals = detect_geometry_and_axis(Pn, verbose=verbose)
    use_mv_eff = bool(use_multivalued) and (str(kind) == "torus")

    if use_mv_eff:
        phi_t, grad_t, phi_p, grad_p = multivalued_bases_about_axis(Pn, Nn, a_hat, verbose=verbose)
        if harmonic_coeffs is None:
            grad_t_bdry, grad_p_bdry = grad_t(Pn), grad_p(Pn)
            a, _D_raw, _D0 = fit_mv_coeffs_minimize_rhs(Nn, W, grad_t_bdry, grad_p_bdry, verbose=verbose)
        else:
            a = jnp.asarray(harmonic_coeffs, dtype=jnp.float64)
        grad_t_bdry, grad_p_bdry = grad_t(Pn), grad_p(Pn)
        g_raw = scinfo.scale * jnp.sum(Nn * (a[0] * grad_t_bdry + a[1] * grad_p_bdry), axis=1)
    else:
        if harmonic_coeffs is not None:
            raise ValueError(f"harmonic_coeffs were provided but geometry_kind={kind!r} is not torus-like.")
        phi_t = phi_p = (lambda Xn: jnp.zeros((Xn.shape[0],), dtype=jnp.float64))
        grad_t = grad_p = (lambda Xn: jnp.zeros_like(Xn, dtype=jnp.float64))
        a = jnp.zeros((2,), dtype=jnp.float64)
        g_raw = jnp.zeros((Pn.shape[0],), dtype=jnp.float64)

    A, _D = build_system_matrices(Pn, Nn, Yn, W, grad_t, grad_p, scinfo, use_mv=use_mv_eff, verbose=verbose)
    alpha = solve_alpha_with_rhs(A, W, g_raw, lam=lambda_reg, verbose=verbose)

    if acceleration == "barnes-hut":
        phi_fn, grad_fn, _psi_fn, _grad_psi_fn, _lap_psi_fn, _grad_mv = build_evaluators_mfs_accel(
            Pn,
            Yn,
            alpha,
            phi_t,
            phi_p,
            a,
            scinfo,
            grad_t,
            grad_p,
            theta=accel_theta,
            leaf_size=accel_leaf_size,
        )
    else:
        phi_fn, grad_fn, _psi_fn, _grad_psi_fn, _lap_psi_fn, _grad_mv = build_evaluators_mfs(
            Pn, Yn, alpha, phi_t, phi_p, a, scinfo, grad_t, grad_p
        )

    metadata = {
        "method": "mfs",
        "use_multivalued": bool(use_mv_eff),
        "k_nn": int(k_nn),
        "source_factor": float(source_factor),
        "lambda_reg": float(lambda_reg),
        "harmonic_coeffs": [float(a[0]), float(a[1])],
        "geometry_kind": str(kind),
        "normals_flipped": bool(flipped),
        "acceleration": str(acceleration),
    }
    return MFSSolution(phi=phi_fn, B=grad_fn, metadata=metadata)
