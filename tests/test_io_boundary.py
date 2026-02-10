import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from bimfx.io import (
    boundary_from_sflm_npy,
    boundary_from_slam_npz,
    boundary_from_stl,
    boundary_from_vmec_wout,
    load_boundary_csv,
    save_boundary_csv,
)


def _write_fake_wout(path: Path) -> None:
    # Minimal VMEC-like file with xm/xn + Fourier coefficients
    with netcdf_file(path, "w") as f:
        f.createDimension("scalar", 1)
        f.createDimension("ns", 3)
        f.createDimension("mn", 2)
        f.createVariable("ns", "i4", ("scalar",))[:] = np.array([3], dtype=int)
        f.createVariable("xm", "i4", ("mn",))[:] = np.array([0, 1], dtype=int)
        f.createVariable("xn", "i4", ("mn",))[:] = np.array([0, 0], dtype=int)

        rmnc = f.createVariable("rmnc", "f8", ("mn", "ns"))
        zmns = f.createVariable("zmns", "f8", ("mn", "ns"))
        # R = R0 + r cos(theta); Z = r sin(theta)
        R0 = 3.0
        r = 1.0
        rmnc[:] = np.array([[R0, R0, R0], [r, r, r]], dtype=float)
        zmns[:] = np.array([[0.0, 0.0, 0.0], [r, r, r]], dtype=float)

        f.createVariable("lasym__logical__", "i4", ("scalar",))[:] = np.array([0], dtype=int)
        f.createVariable("rmns", "f8", ("mn", "ns"))[:] = 0.0
        f.createVariable("zmnc", "f8", ("mn", "ns"))[:] = 0.0


def test_csv_roundtrip(tmp_path: Path) -> None:
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out_stem = tmp_path / "boundary"
    pts_path, nrm_path = save_boundary_csv(points, normals, out_stem)

    data = load_boundary_csv(pts_path, nrm_path)
    assert data.points.shape == (2, 3)
    assert data.normals is not None
    assert np.allclose(data.normals, normals)


def test_vmec_wout_loader(tmp_path: Path) -> None:
    wout_path = tmp_path / "wout_test.nc"
    _write_fake_wout(wout_path)
    data = boundary_from_vmec_wout(wout_path, s=1.0, ntheta=12, nphi=16)
    assert data.points.shape == (12 * 16, 3)
    assert data.normals is not None
    nrm = np.linalg.norm(data.normals, axis=1)
    assert np.allclose(nrm, 1.0, atol=1e-6)


def test_slam_npz_loader(tmp_path: Path) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    theta2d, phi2d = np.meshgrid(theta, phi, indexing="ij")
    R0, r = 3.0, 1.0
    Rg = R0 + r * np.cos(theta2d)
    Zg = r * np.sin(theta2d)
    path = tmp_path / "slam.npz"
    np.savez(path, theta_grid=theta, phi_grid=phi, R_grid=Rg, Z_grid=Zg)

    data = boundary_from_slam_npz(path, compute_normals=True)
    assert data.points.shape == (8 * 10, 3)
    assert data.normals is not None
    nrm = np.linalg.norm(data.normals, axis=1)
    assert np.allclose(nrm, 1.0, atol=1e-6)


def test_sflm_npy_loader(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    P = rng.normal(size=(100, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    N = P.copy()
    arr = np.vstack([P.T, N.T])
    path = tmp_path / "sflm.npy"
    np.save(path, arr)

    data = boundary_from_sflm_npy(path)
    assert data.points.shape == (100, 3)
    assert data.normals is not None


def test_stl_loader_optional(tmp_path: Path) -> None:
    tm = pytest.importorskip("trimesh")
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    stl_path = tmp_path / "sphere.stl"
    mesh.export(stl_path)

    data = boundary_from_stl(stl_path, n_points=200, even=False)
    assert data.points.shape[0] == 200
    assert data.normals is not None
