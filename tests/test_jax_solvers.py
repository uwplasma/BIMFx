import os

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


def _sphere_points(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    P = rng.normal(size=(n, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    N = P.copy()
    return P, N


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


def test_mfs_jax_zero_field_on_sphere():
    _require_x64()
    from bimfx.jax_solvers import solve_mfs_jax

    P, N = _sphere_points(40)
    field = solve_mfs_jax(P, N, k_nn=12, harmonic_coeffs=None)
    Pin = P - 0.05 * N
    B = np.asarray(field.B(Pin))
    assert np.max(np.linalg.norm(B, axis=1)) < 1e-6


def test_mfs_jax_gradients_finite():
    _require_x64()
    import jax
    import jax.numpy as jnp

    from bimfx.jax_solvers import solve_mfs_jax
    from bimfx.objectives import boundary_residual_objective

    P, N = _torus_point_cloud(R=3.0, r=1.0, nphi=6, ntheta=6)
    Pj = jnp.asarray(P)
    Nj = jnp.asarray(N)

    def objective(P_var):
        field = solve_mfs_jax(
            P_var,
            Nj,
            k_nn=12,
            lambda_reg=1e-2,
            harmonic_coeffs=(1.0, 0.0),
            stop_gradient=True,
            a_hat=jnp.array([0.0, 0.0, 1.0]),
        )
        return boundary_residual_objective(field.B, P_var, Nj)

    grad = jax.grad(objective)(Pj)
    assert jnp.isfinite(grad).all()


def test_bim_jax_forward_finite():
    _require_x64()
    import jax.numpy as jnp

    from bimfx.jax_solvers import solve_bim_jax

    P, N = _torus_point_cloud(R=3.0, r=1.0, nphi=6, ntheta=6)
    field = solve_bim_jax(P, N, k_nn=12, lambda_reg=1e-4, clip_factor=0.3, harmonic_coeffs=(1.0, 0.0))
    Pin = jnp.asarray(P - 0.05 * N)
    B = jnp.asarray(field.B(Pin))
    assert jnp.isfinite(B).all()
