from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.sparse.linalg import cg as jax_cg
import numpy as np

from bimfx.mfs.geometry import (
    detect_geometry_and_axis,
    kNN_geometry_stats,
    maybe_flip_normals,
    multivalued_bases_about_axis,
    normalize_geometry,
)


@dataclass(frozen=True)
class BIMSolution:
    phi: Any
    B: Any
    metadata: dict[str, Any]


@jit
def _grad_green_x(diff: jnp.ndarray, r2: jnp.ndarray) -> jnp.ndarray:
    r3 = r2 * jnp.sqrt(r2)
    return -diff / (4.0 * jnp.pi * r3[..., None])


@jit
def build_Kprime(P: jnp.ndarray, N: jnp.ndarray, W: jnp.ndarray, h: jnp.ndarray, *, clip_factor: float) -> jnp.ndarray:
    """Build the K' operator with local near-singular regularization."""
    Xi = P[:, None, :]
    Xj = P[None, :, :]
    diff = Xi - Xj
    r2 = jnp.sum(diff * diff, axis=-1)

    npts = P.shape[0]
    mask = ~jnp.eye(npts, dtype=bool)

    hi = h[:, None]
    hj = h[None, :]
    h_pair = 0.5 * (hi + hj)
    r2_clip = (clip_factor * h_pair) ** 2

    r2_clipped = jnp.maximum(r2, r2_clip)
    r2_safe = jnp.where(mask, r2_clipped, 1.0)

    gradG = _grad_green_x(diff, r2_safe)
    n_dot_grad = jnp.sum(gradG * N[:, None, :], axis=-1)
    n_dot_grad = jnp.where(mask, n_dot_grad, 0.0)

    return n_dot_grad * W[None, :]


@partial(jit, static_argnames=("solver", "preconditioner"))
def solve_density_sigma(
    P: jnp.ndarray,
    N: jnp.ndarray,
    W: jnp.ndarray,
    h: jnp.ndarray,
    g: jnp.ndarray,
    *,
    lambda_reg: float,
    clip_factor: float,
    solver: str = "direct",
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    preconditioner: str = "jacobi",
) -> jnp.ndarray:
    """Solve ``(-1/2 I + K') σ + λ σ = -g`` for σ."""
    npts = P.shape[0]
    Kprime = build_Kprime(P, N, W, h, clip_factor=clip_factor)
    I = jnp.eye(npts, dtype=P.dtype)
    A = -0.5 * I + Kprime + lambda_reg * I
    if solver == "direct":
        return jnp.linalg.solve(A, -g)

    def matvec(x):
        return A @ x

    if preconditioner == "jacobi":
        diag = jnp.diag(A)

        def M(x):
            return x / jnp.maximum(diag, 1e-30)

    else:
        M = None

    sigma, _info = jax_cg(matvec, -g, tol=cg_tol, maxiter=cg_maxiter, M=M)
    return sigma


def make_single_layer_evaluators(
    P_src: jnp.ndarray,
    W_src: jnp.ndarray,
    sigma: jnp.ndarray,
    *,
    h_min: float,
    clip_factor: float,
):
    """Return batched evaluators for φ_s and ∇φ_s for the solved density σ."""
    weight = sigma * W_src
    h2_clip = (clip_factor * h_min) ** 2

    @jit
    def phi_s_at_point(x):
        diff = x[None, :] - P_src
        r = jnp.linalg.norm(diff, axis=-1)
        r = jnp.maximum(r, 1e-12)
        G = 1.0 / (4.0 * jnp.pi * r)
        return jnp.sum(G * weight)

    @jit
    def grad_phi_s_at_point(x):
        diff = x[None, :] - P_src
        r2 = jnp.sum(diff * diff, axis=-1)
        r2 = jnp.maximum(r2, h2_clip)
        gradG = _grad_green_x(diff, r2)
        return jnp.sum(gradG * weight[:, None], axis=0)

    return jit(vmap(phi_s_at_point)), jit(vmap(grad_phi_s_at_point))


def solve_bim_neumann(
    points: Any,
    normals: Any,
    *,
    use_multivalued: bool = True,
    k_nn: int = 48,
    lambda_reg: float = 1e-6,
    clip_factor: float = 0.2,
    harmonic_coeffs: tuple[float, float] | None = None,
    solver: str = "direct",
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    preconditioner: str = "jacobi",
    verbose: bool = True,
) -> BIMSolution:
    """BIM-based Neumann solve enforcing ``n·B = 0`` on the boundary point cloud."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "BIMFx BIM solver requires JAX 64-bit mode for numerical stability. "
            "Set `JAX_ENABLE_X64=1` in your environment (recommended) or call "
            "`jax.config.update('jax_enable_x64', True)` before importing `bimfx`."
        )
    P = jnp.asarray(points, dtype=jnp.float64)
    N = jnp.asarray(normals, dtype=jnp.float64)
    N = N / jnp.maximum(1e-30, jnp.linalg.norm(N, axis=1, keepdims=True))

    N, flipped = maybe_flip_normals(P, N)
    Pn, scinfo = normalize_geometry(P, verbose=verbose)

    Wn, rk_n = kNN_geometry_stats(Pn, k=k_nn, verbose=verbose)
    rk_n = jnp.asarray(rk_n, dtype=jnp.float64)

    # Convert normalized weights/spacings to world units for the BIM kernel.
    W = Wn / (scinfo.scale**2)
    h = rk_n / scinfo.scale
    h_min = float(jnp.min(h))

    kind, a_hat, _E_axes, _c_axes, _svals = detect_geometry_and_axis(Pn, verbose=verbose)
    use_mv_eff = bool(use_multivalued) and (str(kind) == "torus")

    if use_mv_eff:
        _phi_t, grad_t, _phi_p, grad_p = multivalued_bases_about_axis(Pn, N, a_hat, verbose=verbose)
        if harmonic_coeffs is None:
            if verbose:
                print("[BIM] harmonic_coeffs not provided; defaulting to a=(0,0).")
            a = jnp.zeros((2,), dtype=jnp.float64)
        else:
            a = jnp.asarray(harmonic_coeffs, dtype=jnp.float64)
        Gt = grad_t(Pn)
        Gp = grad_p(Pn)
        g = scinfo.scale * jnp.sum(N * (a[0] * Gt + a[1] * Gp), axis=1)
    else:
        if harmonic_coeffs is not None:
            raise ValueError(f"harmonic_coeffs were provided but geometry_kind={kind!r} is not torus-like.")
        a = jnp.zeros((2,), dtype=jnp.float64)
        g = jnp.zeros((P.shape[0],), dtype=jnp.float64)

    sigma = solve_density_sigma(
        P,
        N,
        W,
        h,
        g,
        lambda_reg=lambda_reg,
        clip_factor=clip_factor,
        solver=solver,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        preconditioner=preconditioner,
    )
    phi_s_fn, grad_s_fn = make_single_layer_evaluators(P, W, sigma, h_min=h_min, clip_factor=clip_factor)

    def B_mv_fn(X):
        if not use_mv_eff:
            return jnp.zeros_like(X)
        Xn = (X - scinfo.center) * scinfo.scale
        return scinfo.scale * (a[0] * grad_t(Xn) + a[1] * grad_p(Xn))

    def B_fn(X):
        return B_mv_fn(X) + grad_s_fn(X)

    metadata = {
        "method": "bim",
        "use_multivalued": bool(use_mv_eff),
        "k_nn": int(k_nn),
        "lambda_reg": float(lambda_reg),
        "clip_factor": float(clip_factor),
        "solver": str(solver),
        "harmonic_coeffs": [float(a[0]), float(a[1])],
        "geometry_kind": str(kind),
        "normals_flipped": bool(flipped),
    }
    return BIMSolution(phi=phi_s_fn, B=B_fn, metadata=metadata)
