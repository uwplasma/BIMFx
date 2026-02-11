import os

# Ensure x64 is enabled before JAX is imported anywhere.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest


def _require_x64() -> None:
    import jax

    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
    if not jax.config.jax_enable_x64:
        pytest.skip("JAX 64-bit mode is required for BIMFx solver tests.")


def _torus_point_cloud(R: float, r: float, nphi: int, ntheta: int) -> tuple[np.ndarray, np.ndarray]:
    phis = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    points = []
    normals = []
    for phi in phis:
        cphi, sphi = np.cos(phi), np.sin(phi)
        for th in thetas:
            cth, sth = np.cos(th), np.sin(th)
            x = (R + r * cth) * cphi
            y = (R + r * cth) * sphi
            z = r * sth
            points.append([x, y, z])
            normals.append([cth * cphi, cth * sphi, sth])
    P = np.asarray(points, dtype=float)
    N = np.asarray(normals, dtype=float)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return P, N


def _ellipsoid_point_cloud(a: float, b: float, c: float, nphi: int, ntheta: int) -> tuple[np.ndarray, np.ndarray]:
    phis = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    thetas = np.linspace(0.0, np.pi, ntheta, endpoint=True)
    points = []
    normals = []
    for phi in phis:
        cphi, sphi = np.cos(phi), np.sin(phi)
        for th in thetas:
            sth, cth = np.sin(th), np.cos(th)
            x = a * sth * cphi
            y = b * sth * sphi
            z = c * cth
            nx = x / (a * a)
            ny = y / (b * b)
            nz = z / (c * c)
            n = np.array([nx, ny, nz])
            n /= np.linalg.norm(n)
            points.append([x, y, z])
            normals.append(n.tolist())
    return np.asarray(points, dtype=float), np.asarray(normals, dtype=float)


def test_mfs_blob_returns_zero_field():
    _require_x64()
    from bimfx import solve_mfs

    rng = np.random.default_rng(0)
    P = rng.normal(size=(60, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    N = P.copy()

    field = solve_mfs(P, N, toroidal_flux=None)
    Pin = P - 0.05 * N
    B = np.asarray(field.B(Pin))
    assert np.max(np.linalg.norm(B, axis=1)) < 1e-12


def test_mfs_torus_enforces_tangency():
    _require_x64()
    from bimfx import solve_mfs

    P, N = _torus_point_cloud(R=3.0, r=1.0, nphi=10, ntheta=10)
    field = solve_mfs(P, N, toroidal_flux=1.0)

    Pin = P - 0.05 * N
    B = np.asarray(field.B(Pin))
    ndot = np.sum(N * B, axis=1)
    assert np.sqrt(np.mean(ndot**2)) < 1e-10


def test_bim_torus_enforces_tangency():
    _require_x64()
    from bimfx import solve_bim

    P, N = _torus_point_cloud(R=3.0, r=1.0, nphi=10, ntheta=10)
    field = solve_bim(P, N, toroidal_flux=1.0)

    Pin = P - 0.05 * N
    B = np.asarray(field.B(Pin))
    ndot = np.sum(N * B, axis=1)
    assert np.sqrt(np.mean(ndot**2)) < 1e-10


def test_bim_cg_solver_runs():
    _require_x64()
    from bimfx import solve_bim
    from bimfx.vacuum.solve import SolveOptions

    P, N = _torus_point_cloud(R=3.0, r=1.0, nphi=8, ntheta=8)
    options = SolveOptions(solver="cg", cg_maxiter=200, cg_tol=1e-8, verbose=False)
    field = solve_bim(P, N, toroidal_flux=1.0, options=options)
    Pin = P - 0.05 * N
    B = np.asarray(field.B(Pin))
    assert np.isfinite(B).all()


def test_mfs_ellipsoid_zero_field():
    _require_x64()
    from bimfx import solve_mfs

    P, N = _ellipsoid_point_cloud(2.0, 1.2, 0.8, nphi=16, ntheta=12)
    field = solve_mfs(P, N, toroidal_flux=None)
    Pin = P - 0.05 * N
    B = np.asarray(field.B(Pin))
    assert np.max(np.linalg.norm(B, axis=1)) < 1e-10


def test_bim_ellipsoid_zero_field():
    _require_x64()
    from bimfx import solve_bim

    P, N = _ellipsoid_point_cloud(2.0, 1.2, 0.8, nphi=16, ntheta=12)
    field = solve_bim(P, N, toroidal_flux=None)
    Pin = P - 0.05 * N
    B = np.asarray(field.B(Pin))
    assert np.max(np.linalg.norm(B, axis=1)) < 1e-10


def test_accelerated_evaluators_close_to_direct():
    _require_x64()
    from bimfx import solve_bim, solve_mfs
    from bimfx.vacuum.solve import SolveOptions

    P, N = _torus_point_cloud(R=3.0, r=1.0, nphi=6, ntheta=6)
    Pin = P - 0.05 * N

    opts_ref = SolveOptions(verbose=False)
    opts_accel = SolveOptions(acceleration="barnes-hut", accel_theta=0.6, accel_leaf_size=16, verbose=False)

    field_mfs_ref = solve_mfs(P, N, toroidal_flux=1.0, options=opts_ref)
    field_mfs_fast = solve_mfs(P, N, toroidal_flux=1.0, options=opts_accel)
    B_ref = np.asarray(field_mfs_ref.B(Pin))
    B_fast = np.asarray(field_mfs_fast.B(Pin))
    rel = np.linalg.norm(B_fast - B_ref) / np.maximum(1e-12, np.linalg.norm(B_ref))
    assert rel < 0.2

    field_bim_ref = solve_bim(P, N, toroidal_flux=1.0, options=opts_ref)
    field_bim_fast = solve_bim(P, N, toroidal_flux=1.0, options=opts_accel)
    B_ref = np.asarray(field_bim_ref.B(Pin))
    B_fast = np.asarray(field_bim_fast.B(Pin))
    rel = np.linalg.norm(B_fast - B_ref) / np.maximum(1e-12, np.linalg.norm(B_ref))
    assert rel < 0.2
