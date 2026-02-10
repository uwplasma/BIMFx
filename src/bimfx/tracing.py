from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

Array = Any


@dataclass(frozen=True)
class FieldlineTrace:
    """Container for field-line traces."""

    trajectories: np.ndarray  # (n_seed, n_step+1, 3)
    step: float
    normalize: bool


def trace_fieldlines_rk4(
    B: Callable[[Array], Array],
    seeds: Array,
    *,
    ds: float,
    n_steps: int,
    normalize: bool = True,
) -> FieldlineTrace:
    """Trace field lines with a fixed-step RK4 integrator.

    Solves x'(s) = B(x) (optionally normalized to unit speed).
    """
    seeds = np.asarray(seeds, dtype=float)
    if seeds.ndim == 1:
        seeds = seeds[None, :]
    if seeds.shape[1] != 3:
        raise ValueError(f"Expected seeds shape (N,3); got {seeds.shape}")

    def eval_B(points: np.ndarray) -> np.ndarray:
        try:
            out = np.asarray(B(points))
        except Exception:
            out = np.stack([np.asarray(B(p)) for p in points], axis=0)
        if out.shape == (3,):
            out = out[None, :]
        return out

    def rhs(points: np.ndarray) -> np.ndarray:
        v = eval_B(points)
        if not normalize:
            return v
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(1e-30, nrm)

    n_seed = seeds.shape[0]
    traj = np.empty((n_seed, n_steps + 1, 3), dtype=float)
    traj[:, 0, :] = seeds

    x = seeds.copy()
    for k in range(n_steps):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * ds * k1)
        k3 = rhs(x + 0.5 * ds * k2)
        k4 = rhs(x + ds * k3)
        x = x + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[:, k + 1, :] = x

    return FieldlineTrace(trajectories=traj, step=float(ds), normalize=bool(normalize))


def trace_fieldlines_rk4_jax(
    B: Callable[[Array], Array],
    seeds: Array,
    *,
    ds: float,
    n_steps: int,
    normalize: bool = True,
    return_jax: bool = False,
) -> FieldlineTrace:
    """Trace field lines with RK4 using JAX and `lax.scan`."""
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover
        raise ImportError("JAX is required for the JAX tracing backend.") from exc

    seeds_j = jnp.asarray(seeds, dtype=jnp.float64)
    if seeds_j.ndim == 1:
        seeds_j = seeds_j[None, :]
    if seeds_j.shape[1] != 3:
        raise ValueError(f"Expected seeds shape (N,3); got {seeds_j.shape}")

    def eval_B(points):
        return jnp.asarray(B(points))

    def rhs(points):
        v = eval_B(points)
        if v.shape == (3,):
            v = v[None, :]
        if not normalize:
            return v
        nrm = jnp.linalg.norm(v, axis=1, keepdims=True)
        return v / jnp.maximum(1e-30, nrm)

    def step(x, _):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * ds * k1)
        k3 = rhs(x + 0.5 * ds * k2)
        k4 = rhs(x + ds * k3)
        x_next = x + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next, x_next

    x0 = seeds_j
    _, traj = jax.lax.scan(step, x0, xs=None, length=n_steps)
    traj = jnp.concatenate([x0[None, ...], traj], axis=0)
    traj = jnp.transpose(traj, (1, 0, 2))

    if return_jax:
        traj_out = traj
    else:
        traj_out = np.asarray(traj)

    return FieldlineTrace(trajectories=traj_out, step=float(ds), normalize=bool(normalize))


@dataclass(frozen=True)
class PoincareSection:
    """Poincare R–Z intersection points at specified toroidal angles."""

    R: np.ndarray
    Z: np.ndarray
    seed_index: np.ndarray
    plane_index: np.ndarray
    phi_planes: np.ndarray


def poincare_sections(
    trajectories: np.ndarray,
    *,
    phi_planes: Iterable[float],
    nfp: int = 1,
) -> PoincareSection:
    """Compute Poincare intersections for a set of field-line trajectories.

    Parameters
    ----------
    trajectories:
        Array of shape (n_seed, n_step+1, 3).
    phi_planes:
        Iterable of base plane angles (radians). Each plane is repeated every
        2π/nfp.
    nfp:
        Number of field periods (for toroidal symmetry).
    """
    traj = np.asarray(trajectories, dtype=float)
    if traj.ndim != 3 or traj.shape[2] != 3:
        raise ValueError(f"Expected trajectories shape (N, T, 3); got {traj.shape}")

    period = 2.0 * np.pi / max(1, int(nfp))
    phi_planes = np.asarray(list(phi_planes), dtype=float)

    R_all: list[float] = []
    Z_all: list[float] = []
    seed_idx: list[int] = []
    plane_idx: list[int] = []

    for s_idx in range(traj.shape[0]):
        x = traj[s_idx, :, 0]
        y = traj[s_idx, :, 1]
        z = traj[s_idx, :, 2]
        phi = np.unwrap(np.arctan2(y, x))

        for p_idx, phi0 in enumerate(phi_planes):
            # Find crossings of phi = phi0 + m*period
            for i in range(len(phi) - 1):
                phi_i = phi[i]
                phi_j = phi[i + 1]
                if phi_j == phi_i:
                    continue
                # Determine m range for which the segment crosses the plane
                m_min = int(np.ceil((min(phi_i, phi_j) - phi0) / period))
                m_max = int(np.floor((max(phi_i, phi_j) - phi0) / period))
                for m in range(m_min, m_max + 1):
                    target = phi0 + m * period
                    t = (target - phi_i) / (phi_j - phi_i)
                    if 0.0 <= t <= 1.0:
                        x_t = x[i] + t * (x[i + 1] - x[i])
                        y_t = y[i] + t * (y[i + 1] - y[i])
                        z_t = z[i] + t * (z[i + 1] - z[i])
                        R_all.append(float(np.sqrt(x_t * x_t + y_t * y_t)))
                        Z_all.append(float(z_t))
                        seed_idx.append(s_idx)
                        plane_idx.append(p_idx)

    if len(R_all) == 0:
        R = np.empty((0,), dtype=float)
        Z = np.empty((0,), dtype=float)
        seed_index = np.empty((0,), dtype=int)
        plane_index = np.empty((0,), dtype=int)
    else:
        R = np.asarray(R_all)
        Z = np.asarray(Z_all)
        seed_index = np.asarray(seed_idx, dtype=int)
        plane_index = np.asarray(plane_idx, dtype=int)

    return PoincareSection(R=R, Z=Z, seed_index=seed_index, plane_index=plane_index, phi_planes=phi_planes)
