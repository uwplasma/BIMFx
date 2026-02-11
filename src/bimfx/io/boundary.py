from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import netcdf_file
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class BoundaryData:
    """Boundary point cloud with optional normals and metadata."""

    points: np.ndarray  # (N,3)
    normals: np.ndarray | None
    metadata: dict[str, Any]

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        normals: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "BoundaryData":
        return cls(points=np.asarray(points), normals=normals, metadata=metadata or {})


def _normalize_normals(normals: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(nrm, 1e-30)


def _ensure_outward(points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, bool]:
    center = points.mean(axis=0)
    dots = np.einsum("ij,ij->i", points - center, normals)
    flipped = float(np.mean(dots)) < 0.0
    if flipped:
        normals = -normals
    return normals, flipped


def estimate_normals(points: np.ndarray, *, k: int = 20) -> np.ndarray:
    """Estimate normals from a raw point cloud using local PCA."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3); got {pts.shape}")
    k_eff = min(k + 1, len(pts))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k_eff)
    normals = np.empty_like(pts)
    for i in range(len(pts)):
        neigh = pts[idx[i, 1:]]
        if neigh.shape[0] < 3:
            normals[i] = np.array([0.0, 0.0, 1.0])
            continue
        cov = np.cov((neigh - neigh.mean(axis=0)).T)
        _w, v = np.linalg.eigh(cov)
        n = v[:, 0]
        n = n / np.maximum(np.linalg.norm(n), 1e-30)
        normals[i] = n
    normals, _ = _ensure_outward(pts, normals)
    return _normalize_normals(normals)


def load_boundary_csv(points_path: str | Path, normals_path: str | Path | None = None) -> BoundaryData:
    points = np.loadtxt(points_path, delimiter=",", skiprows=1)
    if points.ndim == 1:
        points = points[None, :]
    if points.shape[1] != 3:
        raise ValueError(f"Expected 3 columns in points CSV, got shape {points.shape}")

    normals = None
    if normals_path is not None:
        normals = np.loadtxt(normals_path, delimiter=",", skiprows=1)
        if normals.ndim == 1:
            normals = normals[None, :]
        if normals.shape != points.shape:
            raise ValueError(f"Normals shape {normals.shape} does not match points shape {points.shape}")
        normals = _normalize_normals(normals)
        normals, flipped = _ensure_outward(points, normals)
    else:
        flipped = False

    return BoundaryData(
        points=points,
        normals=normals,
        metadata={"source": "csv", "normals_flipped": bool(flipped)},
    )


def save_boundary_csv(points: np.ndarray, normals: np.ndarray | None, out_stem: str | Path) -> tuple[Path, Path | None]:
    out_stem = Path(out_stem)
    points_path = out_stem.with_suffix(".csv")
    np.savetxt(points_path, points, delimiter=",", header="x,y,z", comments="")

    normals_path = None
    if normals is not None:
        normals_path = out_stem.with_name(out_stem.name + "_normals").with_suffix(".csv")
        np.savetxt(normals_path, normals, delimiter=",", header="nx,ny,nz", comments="")
    return points_path, normals_path


def boundary_from_vmec_wout(
    wout_path: str | Path,
    *,
    s: float = 1.0,
    ntheta: int = 64,
    nphi: int = 128,
) -> BoundaryData:
    """Compute a boundary point cloud + normals from a VMEC `wout*.nc` file."""
    f = netcdf_file(str(wout_path), "r", mmap=False)
    ns_val = np.asarray(f.variables["ns"][()])
    ns = int(ns_val.item() if ns_val.shape != () else ns_val)
    xn = np.asarray(f.variables["xn"][()])
    xm = np.asarray(f.variables["xm"][()])
    rmnc = np.asarray(f.variables["rmnc"][()])
    zmns = np.asarray(f.variables["zmns"][()])
    lasym_val = np.asarray(f.variables["lasym__logical__"][()])
    lasym = int(lasym_val.item() if lasym_val.shape != () else lasym_val)
    if lasym == 1:
        rmns = np.asarray(f.variables["rmns"][()])
        zmnc = np.asarray(f.variables["zmnc"][()])
    else:
        rmns = np.zeros_like(rmnc)
        zmnc = np.zeros_like(rmnc)

    def _ensure_mnmax_ns(arr: np.ndarray, ns_val: int) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[0] == ns_val and arr.shape[1] != ns_val:
            return arr.T
        return arr

    rmnc = _ensure_mnmax_ns(rmnc, ns)
    rmns = _ensure_mnmax_ns(rmns, ns)
    zmns = _ensure_mnmax_ns(zmns, ns)
    zmnc = _ensure_mnmax_ns(zmnc, ns)

    theta = np.linspace(0.0, 2.0 * np.pi, num=int(ntheta))
    phi = np.linspace(0.0, 2.0 * np.pi, num=int(nphi))
    theta2d, phi2d = np.meshgrid(theta, phi, indexing="ij")

    s_grid = np.linspace(0.0, 1.0, ns)
    rmnc_i = np.array([np.interp(s, s_grid, row) for row in rmnc])
    rmns_i = np.array([np.interp(s, s_grid, row) for row in rmns])
    zmns_i = np.array([np.interp(s, s_grid, row) for row in zmns])
    zmnc_i = np.array([np.interp(s, s_grid, row) for row in zmnc])

    angles = xm[:, None, None] * theta2d[None, :, :] - xn[:, None, None] * phi2d[None, :, :]
    sin_a = np.sin(angles)
    cos_a = np.cos(angles)

    R = np.einsum("m,mjk->jk", rmnc_i, cos_a) + np.einsum("m,mjk->jk", rmns_i, sin_a)
    Z = np.einsum("m,mjk->jk", zmns_i, sin_a) + np.einsum("m,mjk->jk", zmnc_i, cos_a)
    X = R * np.cos(phi2d)
    Y = R * np.sin(phi2d)

    dR_dtheta = np.einsum("m,mjk,m->jk", rmnc_i, -sin_a, xm) + np.einsum("m,mjk,m->jk", rmns_i, cos_a, xm)
    dZ_dtheta = np.einsum("m,mjk,m->jk", zmns_i, cos_a, xm) + np.einsum("m,mjk,m->jk", zmnc_i, -sin_a, xm)
    dR_dphi = np.einsum("m,mjk,m->jk", rmnc_i, sin_a, xn) + np.einsum("m,mjk,m->jk", rmns_i, -cos_a, xn)
    dZ_dphi = np.einsum("m,mjk,m->jk", zmns_i, -cos_a, xn) + np.einsum("m,mjk,m->jk", zmnc_i, sin_a, xn)

    dX_dtheta = dR_dtheta * np.cos(phi2d)
    dY_dtheta = dR_dtheta * np.sin(phi2d)
    dX_dphi = dR_dphi * np.cos(phi2d) - R * np.sin(phi2d)
    dY_dphi = dR_dphi * np.sin(phi2d) + R * np.cos(phi2d)

    g_theta = np.stack([dX_dtheta, dY_dtheta, dZ_dtheta], axis=-1)
    g_phi = np.stack([dX_dphi, dY_dphi, dZ_dphi], axis=-1)
    normal = np.cross(g_theta, g_phi)
    normals = _normalize_normals(normal.reshape(-1, 3))

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    normals, flipped = _ensure_outward(points, normals)

    return BoundaryData(
        points=points,
        normals=normals,
        metadata={
            "source": "vmec_wout",
            "s": float(s),
            "ntheta": int(ntheta),
            "nphi": int(nphi),
            "lasym": bool(lasym),
            "normals_flipped": bool(flipped),
        },
    )


def boundary_from_slam_npz(
    path: str | Path,
    *,
    ntheta: int | None = None,
    nphi: int | None = None,
    compute_normals: bool = True,
) -> BoundaryData:
    """Load SLAM `.npz` with theta/phi grids and R/Z, compute boundary points and normals."""
    nz = np.load(path)
    required = ["theta_grid", "phi_grid", "R_grid", "Z_grid"]
    for key in required:
        if key not in nz:
            raise ValueError(f"Missing key '{key}' in {path}")
    theta = np.asarray(nz["theta_grid"])
    phi = np.asarray(nz["phi_grid"])
    Rg = np.asarray(nz["R_grid"])
    Zg = np.asarray(nz["Z_grid"])

    if theta.ndim != 1 or phi.ndim != 1 or Rg.ndim != 2:
        raise ValueError("Expected 1D theta/phi and 2D R/Z grids in SLAM file.")

    if ntheta is not None or nphi is not None:
        t_out = np.linspace(0.0, 2.0 * np.pi, int(ntheta or theta.size), endpoint=False)
        p_out = np.linspace(0.0, 2.0 * np.pi, int(nphi or phi.size), endpoint=False)
        Rg = _periodic_resample_2d(Rg, theta, phi, t_out, p_out)
        Zg = _periodic_resample_2d(Zg, theta, phi, t_out, p_out)
        theta, phi = t_out, p_out

    theta2d, phi2d = np.meshgrid(theta, phi, indexing="ij")
    X = Rg * np.cos(phi2d)
    Y = Rg * np.sin(phi2d)
    Z = Zg
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    normals = None
    flipped = False
    if compute_normals:
        R_t = _periodic_diff(Rg, theta, axis=0)
        Z_t = _periodic_diff(Zg, theta, axis=0)
        R_p = _periodic_diff(Rg, phi, axis=1)
        Z_p = _periodic_diff(Zg, phi, axis=1)
        dX_dtheta = R_t * np.cos(phi2d)
        dY_dtheta = R_t * np.sin(phi2d)
        dX_dphi = R_p * np.cos(phi2d) - Rg * np.sin(phi2d)
        dY_dphi = R_p * np.sin(phi2d) + Rg * np.cos(phi2d)
        g_theta = np.stack([dX_dtheta, dY_dtheta, Z_t], axis=-1)
        g_phi = np.stack([dX_dphi, dY_dphi, Z_p], axis=-1)
        normal = np.cross(g_theta, g_phi)
        normals = _normalize_normals(normal.reshape(-1, 3))
        normals, flipped = _ensure_outward(points, normals)

    return BoundaryData(
        points=points,
        normals=normals,
        metadata={
            "source": "slam_npz",
            "ntheta": int(theta.size),
            "nphi": int(phi.size),
            "normals_flipped": bool(flipped),
        },
    )


def boundary_from_sflm_npy(path: str | Path) -> BoundaryData:
    """Load SFLM `.npy` with shape (6,N): x,y,z,nx,ny,nz."""
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 2 and arr.shape[0] == 6:
        x, y, z, nx, ny, nz = arr
    elif arr.ndim == 1 and arr.shape[0] == 6:
        x, y, z, nx, ny, nz = list(arr)
    else:
        raise ValueError(f"Expected (6,N) array in {path}, got shape {arr.shape}")

    points = np.vstack([x, y, z]).T
    normals = np.vstack([nx, ny, nz]).T
    normals = _normalize_normals(normals)
    normals, flipped = _ensure_outward(points, normals)

    return BoundaryData(
        points=points,
        normals=normals,
        metadata={"source": "sflm_npy", "normals_flipped": bool(flipped)},
    )


def boundary_from_stl(
    path: str | Path,
    *,
    n_points: int = 2048,
    even: bool = False,
    fix_normals: bool = True,
) -> BoundaryData:
    """Sample a surface mesh from an STL file into points + normals."""
    try:
        import trimesh as tm
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("trimesh is required for STL sampling. Install with `pip install trimesh`.") from exc

    mesh = tm.load(str(path), force="mesh")
    if not isinstance(mesh, tm.Trimesh):
        if isinstance(mesh, tm.Scene) and len(mesh.geometry):
            mesh = tm.util.concatenate(list(mesh.geometry.values()))
        else:
            raise ValueError("Loaded object is not a Trimesh or usable Scene.")

    if fix_normals:
        mesh.rezero()
        try:
            mesh.fix_normals()
        except Exception:
            pass

    if even:
        points = tm.sample.sample_surface_even(mesh, int(n_points))
        _, face_idx = mesh.nearest.on_surface(points)
    else:
        points, face_idx = tm.sample.sample_surface(mesh, int(n_points))

    normals = mesh.face_normals[face_idx]
    normals = _normalize_normals(normals)
    normals, flipped = _ensure_outward(points, normals)

    return BoundaryData(
        points=np.asarray(points),
        normals=np.asarray(normals),
        metadata={"source": "stl", "normals_flipped": bool(flipped), "even": bool(even)},
    )


def boundary_from_mesh(
    path: str | Path,
    *,
    n_points: int = 2048,
    even: bool = False,
    fix_normals: bool = True,
) -> BoundaryData:
    """Sample a surface mesh into points + normals (STL/PLY/OBJ/etc.)."""
    try:
        import trimesh as tm
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("trimesh is required for mesh sampling. Install with `pip install trimesh`.") from exc

    mesh = tm.load_mesh(str(path))
    if not isinstance(mesh, tm.Trimesh):
        mesh = mesh.dump().sum()
    if fix_normals:
        mesh.rezero()
        try:
            mesh.fix_normals()
        except Exception:
            pass
    if even:
        points = tm.sample.sample_surface_even(mesh, int(n_points))
        _, face_idx = mesh.nearest.on_surface(points)
    else:
        points, face_idx = tm.sample.sample_surface(mesh, int(n_points))
    normals = mesh.face_normals[face_idx]
    normals = _normalize_normals(normals)
    normals, flipped = _ensure_outward(points, normals)
    return BoundaryData(
        points=np.asarray(points),
        normals=np.asarray(normals),
        metadata={"source": "mesh", "normals_flipped": bool(flipped), "even": bool(even)},
    )


def load_boundary(
    path: str | Path,
    *,
    normals_path: str | Path | None = None,
    format: str | None = None,
    estimate_normals: bool = False,
    normal_k: int = 20,
    n_points: int = 2048,
    even: bool = False,
) -> BoundaryData:
    """Load boundary data with format autodetection."""
    path = Path(path)
    fmt = (format or path.suffix.lstrip(".")).lower()

    if fmt in {"csv"}:
        data = load_boundary_csv(path, normals_path)
        if data.normals is None and estimate_normals:
            normals = estimate_normals(data.points, k=normal_k)
            return BoundaryData(
                points=data.points,
                normals=normals,
                metadata={**data.metadata, "normals_estimated": True},
            )
        return data
    if fmt in {"nc"}:
        return boundary_from_vmec_wout(path)
    if fmt in {"npz"}:
        return boundary_from_slam_npz(path)
    if fmt in {"npy"}:
        return boundary_from_sflm_npy(path)
    if fmt in {"stl", "ply", "obj"}:
        return boundary_from_mesh(path, n_points=n_points, even=even)
    raise ValueError(f"Unsupported boundary format: {fmt}")


def _periodic_diff(values: np.ndarray, coord: np.ndarray, axis: int) -> np.ndarray:
    coord = np.unwrap(np.asarray(coord, float))
    values = np.asarray(values)
    v_p = np.roll(values, -1, axis=axis)
    v_m = np.roll(values, 1, axis=axis)
    c_p = np.roll(coord, -1)
    c_m = np.roll(coord, 1)
    denom = c_p - c_m
    denom = np.where(denom == 0.0, 1e-15, denom)
    shape = [1] * values.ndim
    shape[axis] = coord.size
    return (v_p - v_m) / denom.reshape(shape)


def _periodic_resample_2d(
    values: np.ndarray,
    theta_in: np.ndarray,
    phi_in: np.ndarray,
    theta_out: np.ndarray,
    phi_out: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values)
    theta_in = np.unwrap(np.asarray(theta_in, float))
    phi_in = np.unwrap(np.asarray(phi_in, float))
    theta_out = np.unwrap(np.asarray(theta_out, float))
    phi_out = np.unwrap(np.asarray(phi_out, float))

    v_tiled = np.concatenate([values, values, values], axis=0)
    t_tiled = np.concatenate([theta_in - 2 * np.pi, theta_in, theta_in + 2 * np.pi])
    interp_theta = np.empty((theta_out.size, values.shape[1]), dtype=values.dtype)
    for j in range(values.shape[1]):
        interp_theta[:, j] = np.interp(theta_out, t_tiled, v_tiled[:, j])

    v_tiled_p = np.concatenate([interp_theta, interp_theta, interp_theta], axis=1)
    p_tiled = np.concatenate([phi_in - 2 * np.pi, phi_in, phi_in + 2 * np.pi])
    out = np.empty((theta_out.size, phi_out.size), dtype=values.dtype)
    for i in range(theta_out.size):
        out[i, :] = np.interp(phi_out, p_tiled, v_tiled_p[i, :])
    return out
