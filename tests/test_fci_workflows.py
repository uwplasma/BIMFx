import numpy as np

import pytest

from bimfx.fci import analyze_flux_surfaces, fit_flux_surfaces


def _sphere_points(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    P = rng.normal(size=(n, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    N = P.copy()
    return P, N


def _B_helical(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.stack([-y, x, 0.2 * np.ones_like(z)], axis=1)


def test_fit_flux_surfaces_sphere():
    pytest.importorskip("skimage")
    xs = np.linspace(-1.0, 1.0, 40)
    ys = np.linspace(-1.0, 1.0, 40)
    zs = np.linspace(-1.0, 1.0, 40)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    psi = X**2 + Y**2 + Z**2

    surfaces = fit_flux_surfaces(psi, xs, ys, zs, levels=[0.25])
    assert len(surfaces) == 1
    r = np.linalg.norm(surfaces[0].vertices, axis=1)
    assert np.median(r) == pytest.approx(0.5, abs=0.08)


def test_analyze_flux_surfaces_shapes():
    xs = np.linspace(-1.0, 1.0, 32)
    ys = np.linspace(-1.0, 1.0, 32)
    zs = np.linspace(-1.0, 1.0, 32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    psi = X**2 + Y**2 + Z**2

    P, N = _sphere_points(300)
    analysis = analyze_flux_surfaces(
        _B_helical,
        psi,
        xs,
        ys,
        zs,
        P=P,
        N=N,
        n_seed=6,
        ds=0.03,
        n_steps=300,
        poincare_phi_planes=[0.0, 0.7],
        slice_n_r=48,
        slice_n_z=48,
    )

    assert analysis.psi_along.shape[0] == 6
    assert analysis.psi_along.shape[1] == 301
    assert len(analysis.psi_slices) == 2
    assert analysis.psi_slices[0].psi.shape == (48, 48)
