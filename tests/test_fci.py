import numpy as np

from bimfx.fci import solve_flux_psi_fci


def _sphere_points(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    P = rng.normal(size=(n, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    N = P.copy()
    return P, N


def _B_uniform(points: np.ndarray) -> np.ndarray:
    # Constant field along z
    return np.tile(np.array([0.0, 0.0, 1.0]), (points.shape[0], 1))


def test_fci_solver_boundary_axis_bands():
    P, N = _sphere_points(300)
    sol = solve_flux_psi_fci(
        _B_uniform,
        P,
        N,
        nx=24,
        ny=24,
        nz=24,
        pad_frac=0.0,
        boundary_band_frac=0.08,
        axis_band_frac=0.08,
        maxiter=300,
    )

    # Boundary band should be near 1
    psi_b = sol.psi[sol.boundary_band]
    assert np.allclose(np.median(psi_b), 1.0, atol=1e-2)

    # Axis band should be near 0
    psi_a = sol.psi[sol.axis_band]
    assert np.allclose(np.median(psi_a), 0.0, atol=1e-2)

