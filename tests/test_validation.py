import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from bimfx.validation import boundary_normal_residual, divergence_on_grid


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


def test_boundary_residual_zero_field():
    def B(points: np.ndarray) -> np.ndarray:
        return np.zeros_like(points)

    rng = np.random.default_rng(0)
    P = rng.normal(size=(50, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    N = P.copy()
    res = boundary_normal_residual(B, P, N, normalize=False)
    assert np.allclose(res, 0.0)


def test_divergence_uniform_field_is_zero():
    def B(points: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0.0, 0.0, 1.0]), (points.shape[0], 1))

    xs = np.linspace(-1.0, 1.0, 12)
    ys = np.linspace(-1.0, 1.0, 12)
    zs = np.linspace(-1.0, 1.0, 12)
    divB = divergence_on_grid(B, xs, ys, zs)
    assert np.max(np.abs(divB)) < 1e-12


def test_solver_boundary_residual_small():
    _require_x64()
    from bimfx.vacuum.solve import SolveOptions
    from bimfx import solve_mfs, solve_bim

    P, N = _torus_point_cloud(R=3.0, r=1.0, nphi=12, ntheta=12)
    options = SolveOptions(k_nn=24, verbose=False)
    field_mfs = solve_mfs(P, N, toroidal_flux=1.0, options=options)
    field_bim = solve_bim(P, N, toroidal_flux=1.0, options=options)

    res_mfs = boundary_normal_residual(field_mfs.B, P, N, normalize=True)
    res_bim = boundary_normal_residual(field_bim.B, P, N, normalize=True)
    assert np.sqrt(np.mean(res_mfs**2)) < 1e-8
    assert np.sqrt(np.mean(res_bim**2)) < 1e-8
