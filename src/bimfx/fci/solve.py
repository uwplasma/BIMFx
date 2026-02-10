from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg

Array = Any


@dataclass(frozen=True)
class FCISolution:
    psi: np.ndarray  # (nx, ny, nz)
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    inside: np.ndarray
    boundary_band: np.ndarray
    axis_band: np.ndarray
    metadata: dict[str, Any]


def solve_flux_psi_fci(
    B: Callable[[Array], Array],
    boundary_points: np.ndarray,
    boundary_normals: np.ndarray,
    *,
    nx: int = 48,
    ny: int = 48,
    nz: int = 48,
    pad_frac: float = 0.1,
    eps_perp: float = 1e-3,
    boundary_band_frac: float = 0.02,
    axis_band_frac: float = 0.03,
    maxiter: int = 500,
    tol: float = 1e-8,
) -> FCISolution:
    """Solve for a flux-like scalar psi using anisotropic diffusion.

    The solver builds a Cartesian grid, aligns diffusion with the magnetic
    field direction, and enforces psi=1 on a boundary band and psi=0 on
    an axis band.
    """
    P = np.asarray(boundary_points, dtype=float)
    N = np.asarray(boundary_normals, dtype=float)
    N = N / np.maximum(1e-30, np.linalg.norm(N, axis=1, keepdims=True))

    # Grid
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    span = maxs - mins
    mins = mins - pad_frac * span
    maxs = maxs + pad_frac * span
    xs = np.linspace(mins[0], maxs[0], nx)
    ys = np.linspace(mins[1], maxs[1], ny)
    zs = np.linspace(mins[2], maxs[2], nz)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # Inside mask via nearest boundary normal
    tree = cKDTree(P)
    _, idx = tree.query(grid_pts, k=1)
    Pn = P[idx]
    Nn = N[idx]
    signed = np.einsum("ij,ij->i", grid_pts - Pn, Nn)
    inside = signed < 0.0
    inside3 = inside.reshape(nx, ny, nz)

    # Boundary band: near the surface
    band_thickness = boundary_band_frac * np.max(span)
    boundary_band = (np.abs(signed) <= band_thickness) & inside
    boundary_band3 = boundary_band.reshape(nx, ny, nz)

    # Axis band: near PCA axis
    center = P.mean(axis=0)
    Xc = P - center
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    a_hat = vt[-1]  # smallest variance direction for torus-like boundary
    a_hat = a_hat / (np.linalg.norm(a_hat) + 1e-30)

    r_vec = grid_pts - center[None, :]
    r_par = np.einsum("ij,j->i", r_vec, a_hat)[:, None] * a_hat[None, :]
    r_perp = r_vec - r_par
    dist_axis = np.linalg.norm(r_perp, axis=1)
    axis_radius = axis_band_frac * np.max(span)
    axis_band = (dist_axis <= axis_radius) & inside
    axis_band3 = axis_band.reshape(nx, ny, nz)

    # Free nodes: inside but not boundary/axis bands
    free = inside & (~boundary_band) & (~axis_band)
    free_idx = np.where(free)[0]
    n_free = free_idx.size
    if n_free == 0:
        raise RuntimeError("No free nodes to solve; adjust bands or grid resolution.")

    # Evaluate B and build diffusion tensor diagonal entries
    Bv = np.asarray(B(grid_pts))
    if Bv.shape == (3,):
        Bv = Bv[None, :]
    Bn = Bv / np.maximum(1e-30, np.linalg.norm(Bv, axis=1, keepdims=True))
    tx, ty, tz = Bn[:, 0], Bn[:, 1], Bn[:, 2]
    Dxx = eps_perp + (1.0 - eps_perp) * (tx * tx)
    Dyy = eps_perp + (1.0 - eps_perp) * (ty * ty)
    Dzz = eps_perp + (1.0 - eps_perp) * (tz * tz)

    # Build sparse system A psi = b for free nodes
    # 7-point stencil with variable diagonal diffusion (no cross terms)
    index_map = -np.ones(grid_pts.shape[0], dtype=int)
    index_map[free_idx] = np.arange(n_free)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(n_free, dtype=float)

    def add_entry(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    def handle_neighbor(i_free: int, j_flat: int, coeff: float) -> None:
        if free[j_flat]:
            add_entry(i_free, index_map[j_flat], -coeff)
        else:
            # Dirichlet nodes: psi=1 on boundary band, psi=0 on axis band, psi=0 outside
            if boundary_band[j_flat]:
                b[i_free] += coeff * 1.0
            else:
                b[i_free] += 0.0

    for flat in free_idx:
        i = index_map[flat]
        ix = flat // (ny * nz)
        iy = (flat // nz) % ny
        iz = flat % nz

        # central coefficient
        cx = Dxx[flat] / (dx * dx)
        cy = Dyy[flat] / (dy * dy)
        cz = Dzz[flat] / (dz * dz)
        diag = 2.0 * (cx + cy + cz)
        add_entry(i, i, diag)

        # neighbors
        if ix > 0:
            handle_neighbor(i, flat - ny * nz, cx)
        if ix < nx - 1:
            handle_neighbor(i, flat + ny * nz, cx)
        if iy > 0:
            handle_neighbor(i, flat - nz, cy)
        if iy < ny - 1:
            handle_neighbor(i, flat + nz, cy)
        if iz > 0:
            handle_neighbor(i, flat - 1, cz)
        if iz < nz - 1:
            handle_neighbor(i, flat + 1, cz)

    A = coo_matrix((vals, (rows, cols)), shape=(n_free, n_free)).tocsr()
    sol, info = cg(A, b, rtol=tol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info}).")

    psi = np.zeros(grid_pts.shape[0], dtype=float)
    psi[boundary_band] = 1.0
    psi[axis_band] = 0.0
    psi[free_idx] = sol
    psi3 = psi.reshape(nx, ny, nz)

    return FCISolution(
        psi=psi3,
        xs=xs,
        ys=ys,
        zs=zs,
        inside=inside3,
        boundary_band=boundary_band3,
        axis_band=axis_band3,
        metadata={
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "eps_perp": float(eps_perp),
            "boundary_band_frac": float(boundary_band_frac),
            "axis_band_frac": float(axis_band_frac),
        },
    )
