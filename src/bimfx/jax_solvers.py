from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from bimfx.mfs.geometry import grad_azimuth_about_axis


Array = Any

__all__ = ["solve_mfs_jax", "solve_bim_jax", "MFSJaxField", "BIMJaxField"]

def _normalize_geometry_jax(P: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    center = jnp.mean(P, axis=0)
    r = jnp.linalg.norm(P - center, axis=1)
    r_med = jnp.median(r)
    scale = 1.0 / jnp.maximum(r_med, 1e-12)
    Pn = (P - center) * scale
    return Pn, center, scale


def _knn_radius_jax(P: jnp.ndarray, k: int) -> jnp.ndarray:
    n = P.shape[0]
    diff = P[:, None, :] - P[None, :, :]
    r2 = jnp.sum(diff * diff, axis=-1)
    r2 = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, r2)
    r = jnp.sqrt(r2)
    k_eff = jnp.minimum(k, n - 1)
    r_sorted = jnp.sort(r, axis=1)
    return r_sorted[:, k_eff - 1]


def _solve_alpha_ls(A: jnp.ndarray, W: jnp.ndarray, g: jnp.ndarray, lam: float) -> jnp.ndarray:
    Wsqrt = jnp.sqrt(W)
    Aw = Wsqrt[:, None] * A
    gw = Wsqrt * g
    ATA = Aw.T @ Aw
    ATg = Aw.T @ gw
    NE = ATA + (lam**2) * jnp.eye(A.shape[1], dtype=A.dtype)
    rhs = -ATg
    L = jnp.linalg.cholesky(NE)
    y = jsp_linalg.solve_triangular(L, rhs, lower=True)
    alpha = jsp_linalg.solve_triangular(L.T, y, lower=False)
    return alpha


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MFSJaxField:
    center: jnp.ndarray
    scale: jnp.ndarray
    Yn: jnp.ndarray
    alpha: jnp.ndarray
    a_t: jnp.ndarray
    a_hat: jnp.ndarray

    def tree_flatten(self):
        return ((self.center, self.scale, self.Yn, self.alpha, self.a_t, self.a_hat), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        center, scale, Yn, alpha, a_t, a_hat = children
        return cls(center=center, scale=scale, Yn=Yn, alpha=alpha, a_t=a_t, a_hat=a_hat)

    def phi(self, X: Array) -> Array:
        Xn = (jnp.asarray(X) - self.center) * self.scale
        diff = Xn[:, None, :] - self.Yn[None, :, :]
        r = jnp.linalg.norm(diff, axis=-1)
        G = 1.0 / (4.0 * jnp.pi * jnp.maximum(1e-30, r))
        return G @ self.alpha

    def B(self, X: Array) -> Array:
        Xn = (jnp.asarray(X) - self.center) * self.scale
        diff = Xn[:, None, :] - self.Yn[None, :, :]
        r2 = jnp.sum(diff * diff, axis=-1)
        r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
        gradG = -diff / (4.0 * jnp.pi * r3[..., None])
        grad_s = self.scale * jnp.sum(gradG * self.alpha[None, :, None], axis=1)
        grad_t = self.scale * self.a_t * grad_azimuth_about_axis(Xn, self.a_hat)
        return grad_s + grad_t


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BIMJaxField:
    P: jnp.ndarray
    W: jnp.ndarray
    sigma: jnp.ndarray
    clip_factor: jnp.ndarray
    h_min: jnp.ndarray
    center: jnp.ndarray
    scale: jnp.ndarray
    a_t: jnp.ndarray
    a_hat: jnp.ndarray

    def tree_flatten(self):
        return (
            (
                self.P,
                self.W,
                self.sigma,
                jnp.asarray(self.clip_factor),
                jnp.asarray(self.h_min),
                self.center,
                self.scale,
                self.a_t,
                self.a_hat,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        P, W, sigma, clip_factor, h_min, center, scale, a_t, a_hat = children
        return cls(
            P=P,
            W=W,
            sigma=sigma,
            clip_factor=jnp.asarray(clip_factor),
            h_min=jnp.asarray(h_min),
            center=center,
            scale=scale,
            a_t=a_t,
            a_hat=a_hat,
        )

    def B(self, X: Array) -> Array:
        X = jnp.asarray(X)
        diff = X[:, None, :] - self.P[None, :, :]
        r2 = jnp.sum(diff * diff, axis=-1)
        r2 = jnp.maximum(r2, (self.clip_factor * self.h_min) ** 2)
        gradG = -diff / (4.0 * jnp.pi * r2[..., None] * jnp.sqrt(r2)[..., None])
        weight = self.sigma * self.W
        grad_s = jnp.sum(gradG * weight[None, :, None], axis=1)
        Xn = (X - self.center) * self.scale
        grad_t = self.scale * self.a_t * grad_azimuth_about_axis(Xn, self.a_hat)
        return grad_s + grad_t


def solve_mfs_jax(
    points: Array,
    normals: Array,
    *,
    k_nn: int = 32,
    source_factor: float = 2.0,
    lambda_reg: float = 1e-6,
    harmonic_coeffs: tuple[float, float] | None = None,
    a_hat: Array | None = None,
    stop_gradient: bool = False,
) -> MFSJaxField:
    """JAX-native MFS solve (dense, differentiable)."""
    P = jnp.asarray(points, dtype=jnp.float64)
    N = jnp.asarray(normals, dtype=jnp.float64)
    N = N / jnp.maximum(1e-30, jnp.linalg.norm(N, axis=1, keepdims=True))

    Pn, center, scale = _normalize_geometry_jax(P)
    rk = jax.lax.stop_gradient(_knn_radius_jax(Pn, k_nn))
    W = jnp.pi * rk**2
    Yn = Pn + source_factor * rk[:, None] * N

    diff = Pn[:, None, :] - Yn[None, :, :]
    r2 = jnp.sum(diff * diff, axis=-1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    gradG = -diff / (4.0 * jnp.pi * r3[..., None])
    A = scale * jnp.sum(gradG * N[:, None, :], axis=-1)

    if harmonic_coeffs is None:
        a_t = jnp.array(0.0, dtype=jnp.float64)
        g_raw = jnp.zeros((P.shape[0],), dtype=jnp.float64)
        a_hat_use = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
    else:
        a_t = jnp.array(float(harmonic_coeffs[0]), dtype=jnp.float64)
        if a_hat is None:
            # axis from PCA: smallest singular vector
            _, _, vt = jnp.linalg.svd(Pn - jnp.mean(Pn, axis=0), full_matrices=False)
            a_hat_use = vt[-1]
        else:
            a_hat_use = jnp.asarray(a_hat, dtype=jnp.float64)
        grad_t = grad_azimuth_about_axis(Pn, a_hat_use)
        g_raw = scale * jnp.sum(N * (a_t * grad_t), axis=1)

    alpha = _solve_alpha_ls(A, W, g_raw, lambda_reg)
    if stop_gradient:
        alpha = jax.lax.stop_gradient(alpha)
    return MFSJaxField(
        center=center,
        scale=scale,
        Yn=Yn,
        alpha=alpha,
        a_t=a_t,
        a_hat=jnp.asarray(a_hat_use, dtype=jnp.float64),
    )


def solve_bim_jax(
    points: Array,
    normals: Array,
    *,
    k_nn: int = 32,
    lambda_reg: float = 1e-6,
    clip_factor: float = 0.2,
    harmonic_coeffs: tuple[float, float] | None = None,
    a_hat: Array | None = None,
    stop_gradient: bool = False,
) -> BIMJaxField:
    """JAX-native BIM solve (dense, differentiable)."""
    P = jnp.asarray(points, dtype=jnp.float64)
    N = jnp.asarray(normals, dtype=jnp.float64)
    N = N / jnp.maximum(1e-30, jnp.linalg.norm(N, axis=1, keepdims=True))

    Pn, center, scale = _normalize_geometry_jax(P)
    rk_n = jax.lax.stop_gradient(_knn_radius_jax(Pn, k_nn))
    Wn = jnp.pi * rk_n**2
    W = Wn / (scale**2)
    h = rk_n / scale
    h_min = jnp.min(h)

    diff = P[:, None, :] - P[None, :, :]
    r2 = jnp.sum(diff * diff, axis=-1)
    npts = P.shape[0]
    mask = ~jnp.eye(npts, dtype=bool)

    hi = h[:, None]
    hj = h[None, :]
    h_pair = 0.5 * (hi + hj)
    r2_clip = (clip_factor * h_pair) ** 2
    r2_clipped = jnp.maximum(r2, r2_clip)
    r2_safe = jnp.where(mask, r2_clipped, 1.0)
    gradG = -diff / (4.0 * jnp.pi * r2_safe[..., None] * jnp.sqrt(r2_safe)[..., None])
    n_dot_grad = jnp.sum(gradG * N[:, None, :], axis=-1)
    n_dot_grad = jnp.where(mask, n_dot_grad, 0.0)
    Kprime = n_dot_grad * W[None, :]

    if harmonic_coeffs is None:
        a_t = jnp.array(0.0, dtype=jnp.float64)
        g = jnp.zeros((P.shape[0],), dtype=jnp.float64)
        a_hat_use = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
    else:
        a_t = jnp.array(float(harmonic_coeffs[0]), dtype=jnp.float64)
        if a_hat is None:
            _, _, vt = jnp.linalg.svd(Pn - jnp.mean(Pn, axis=0), full_matrices=False)
            a_hat_use = vt[-1]
        else:
            a_hat_use = jnp.asarray(a_hat, dtype=jnp.float64)
        grad_t = grad_azimuth_about_axis(Pn, a_hat_use)
        g = scale * jnp.sum(N * (a_t * grad_t), axis=1)

    A = -0.5 * jnp.eye(npts, dtype=P.dtype) + Kprime + lambda_reg * jnp.eye(npts, dtype=P.dtype)
    sigma = jnp.linalg.solve(A, -g)
    if stop_gradient:
        sigma = jax.lax.stop_gradient(sigma)
    return BIMJaxField(
        P=P,
        W=W,
        sigma=sigma,
        clip_factor=jnp.asarray(clip_factor),
        h_min=jnp.asarray(h_min),
        center=center,
        scale=scale,
        a_t=a_t,
        a_hat=jnp.asarray(a_hat_use, dtype=jnp.float64),
    )
