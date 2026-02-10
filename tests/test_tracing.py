import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from bimfx.tracing import poincare_sections, trace_fieldlines_rk4, trace_fieldlines_rk4_jax


def _rotational_field(points: np.ndarray) -> np.ndarray:
    # B = (-y, x, 0) -> circles in xy
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.stack([-y, x, 0.0 * z], axis=1)


def _rotational_field_jax(points):
    import jax.numpy as jnp

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return jnp.stack([-y, x, 0.0 * z], axis=1)


def test_trace_fieldlines_circle_returns_close_to_start():
    seed = np.array([[1.0, 0.0, 0.0]])
    n_steps = int(2 * np.pi / 0.05)
    trace = trace_fieldlines_rk4(_rotational_field, seed, ds=0.05, n_steps=n_steps, normalize=True)
    traj = trace.trajectories[0]
    start = traj[0]
    end = traj[-1]
    # After ~2π, should be near start
    dist = np.linalg.norm(end - start)
    assert dist < 0.2


def test_trace_fieldlines_jax_matches_numpy():
    seed = np.array([[1.0, 0.0, 0.0]])
    n_steps = 200
    trace_np = trace_fieldlines_rk4(_rotational_field, seed, ds=0.05, n_steps=n_steps, normalize=True)
    trace_jx = trace_fieldlines_rk4_jax(_rotational_field_jax, seed, ds=0.05, n_steps=n_steps, normalize=True)
    assert np.allclose(trace_np.trajectories, trace_jx.trajectories, atol=1e-5)


def test_poincare_sections_circle():
    seed = np.array([[1.0, 0.0, 0.0]])
    trace = trace_fieldlines_rk4(_rotational_field, seed, ds=0.05, n_steps=400, normalize=True)
    sec = poincare_sections(trace.trajectories, phi_planes=[0.0], nfp=1)
    assert sec.R.size > 0
    # Intersections should lie near R=1, Z=0
    assert np.allclose(sec.R, 1.0, atol=0.1)
    assert np.allclose(sec.Z, 0.0, atol=1e-3)
