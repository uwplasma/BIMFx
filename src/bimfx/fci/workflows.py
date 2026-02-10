from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from bimfx.fci.surfaces import extract_isosurfaces
from bimfx.tracing import (
    FieldlineTrace,
    PoincareSection,
    poincare_sections,
    trace_fieldlines_rk4,
    trace_fieldlines_rk4_jax,
)


@dataclass(frozen=True)
class FluxSurface:
    """Container for a fitted flux surface."""

    level: float
    vertices: np.ndarray
    faces: np.ndarray
    centroid: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray


@dataclass(frozen=True)
class PsiSlice:
    """R–Z slice of psi at a fixed toroidal angle."""

    phi: float
    R: np.ndarray
    Z: np.ndarray
    psi: np.ndarray


@dataclass(frozen=True)
class FluxSurfaceAnalysis:
    """Aggregate results for flux-surface analysis."""

    trace: FieldlineTrace
    poincare: PoincareSection
    s: np.ndarray
    psi_along: np.ndarray
    psi_slices: list[PsiSlice]


def fit_flux_surfaces(
    psi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    levels: Iterable[float],
) -> list[FluxSurface]:
    """Fit flux surfaces by extracting psi isosurfaces."""
    surfaces = extract_isosurfaces(psi, xs, ys, zs, levels=levels)
    out: list[FluxSurface] = []
    for surf in surfaces:
        verts = np.asarray(surf["vertices"])
        faces = np.asarray(surf["faces"])
        centroid = np.mean(verts, axis=0)
        bounds_min = np.min(verts, axis=0)
        bounds_max = np.max(verts, axis=0)
        out.append(
            FluxSurface(
                level=float(surf["level"]),
                vertices=verts,
                faces=faces,
                centroid=centroid,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
            )
        )
    return out


def seed_from_boundary(
    P: np.ndarray,
    N: np.ndarray | None = None,
    *,
    n_seed: int = 16,
    inward_frac: float = 0.02,
) -> np.ndarray:
    """Seed points slightly inside a boundary point cloud."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected P shape (N,3); got {P.shape}")
    center = np.mean(P, axis=0)
    radius = np.median(np.linalg.norm(P - center[None, :], axis=1))
    eps = inward_frac * max(radius, 1e-12)

    if N is None:
        dirs = P - center[None, :]
        dirs /= np.maximum(1e-30, np.linalg.norm(dirs, axis=1, keepdims=True))
        seeds = P - eps * dirs
    else:
        N = np.asarray(N, dtype=float)
        seeds = P - eps * N

    stride = max(1, seeds.shape[0] // max(1, n_seed))
    return seeds[::stride][:n_seed]


def _rz_bounds(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    P: np.ndarray | None = None,
    *,
    margin: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if P is None:
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        R = np.sqrt(XX**2 + YY**2)
        rmin, rmax = float(np.min(R)), float(np.max(R))
        zmin, zmax = float(np.min(zs)), float(np.max(zs))
    else:
        P = np.asarray(P, dtype=float)
        R = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
        rmin, rmax = float(np.min(R)), float(np.max(R))
        zmin, zmax = float(np.min(P[:, 2])), float(np.max(P[:, 2]))

    dr = (rmax - rmin) * margin
    dz = (zmax - zmin) * margin
    return (rmin - dr, rmax + dr), (zmin - dz, zmax + dz)


def compute_psi_rz_slices(
    psi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    phi_planes: Sequence[float],
    *,
    P: np.ndarray | None = None,
    n_r: int = 200,
    n_z: int = 200,
    margin: float = 0.05,
) -> list[PsiSlice]:
    """Compute psi(R,Z) slices at fixed toroidal angles."""
    interp = RegularGridInterpolator((xs, ys, zs), psi, bounds_error=False, fill_value=np.nan)
    (rmin, rmax), (zmin, zmax) = _rz_bounds(xs, ys, zs, P=P, margin=margin)

    R = np.linspace(rmin, rmax, n_r)
    Z = np.linspace(zmin, zmax, n_z)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    out: list[PsiSlice] = []
    for phi in phi_planes:
        X = RR * np.cos(phi)
        Y = RR * np.sin(phi)
        pts = np.stack([X, Y, ZZ], axis=-1).reshape(-1, 3)
        vals = interp(pts).reshape(n_z, n_r)
        out.append(PsiSlice(phi=float(phi), R=R, Z=Z, psi=vals))
    return out


def _sample_psi_on_trajectories(
    psi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    trace: FieldlineTrace,
) -> tuple[np.ndarray, np.ndarray]:
    interp = RegularGridInterpolator((xs, ys, zs), psi, bounds_error=False, fill_value=np.nan)
    traj = np.asarray(trace.trajectories, dtype=float)
    n_seed, n_step, _ = traj.shape
    vals = interp(traj.reshape(-1, 3)).reshape(n_seed, n_step)
    s = np.arange(n_step, dtype=float) * trace.step
    return s, vals


def analyze_flux_surfaces(
    B: Callable[[Any], Any],
    psi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    P: np.ndarray | None = None,
    N: np.ndarray | None = None,
    seeds: np.ndarray | None = None,
    n_seed: int = 16,
    poincare_phi_planes: Sequence[float] = (0.0,),
    poincare_nfp: int = 1,
    slice_phi_planes: Sequence[float] | None = None,
    ds: float = 0.02,
    n_steps: int = 2000,
    normalize: bool = True,
    slice_n_r: int = 200,
    slice_n_z: int = 200,
    slice_margin: float = 0.05,
    trace_backend: str = "numpy",
) -> FluxSurfaceAnalysis:
    """Run a full analysis pass: field-line trace, Poincare, psi slices."""
    if seeds is None:
        if P is None:
            raise ValueError("Provide seeds or boundary points P for default seeding.")
        seeds = seed_from_boundary(P, N, n_seed=n_seed)

    if trace_backend == "jax":
        trace = trace_fieldlines_rk4_jax(B, seeds, ds=ds, n_steps=n_steps, normalize=normalize)
    else:
        trace = trace_fieldlines_rk4(B, seeds, ds=ds, n_steps=n_steps, normalize=normalize)
    poincare = poincare_sections(
        trace.trajectories, phi_planes=poincare_phi_planes, nfp=poincare_nfp
    )
    s, psi_along = _sample_psi_on_trajectories(psi, xs, ys, zs, trace)

    if slice_phi_planes is None:
        slice_phi_planes = tuple(float(p) for p in poincare_phi_planes)

    slices = compute_psi_rz_slices(
        psi,
        xs,
        ys,
        zs,
        slice_phi_planes,
        P=P,
        n_r=slice_n_r,
        n_z=slice_n_z,
        margin=slice_margin,
    )

    return FluxSurfaceAnalysis(
        trace=trace,
        poincare=poincare,
        s=s,
        psi_along=psi_along,
        psi_slices=slices,
    )
