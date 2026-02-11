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


def _finite_diff_grad(fun, A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Ap = A.copy()
            Am = A.copy()
            Ap[i, j] += eps
            Am[i, j] -= eps
            grad[i, j] = (fun(Ap) - fun(Am)) / (2 * eps)
    return grad


def test_solve_linear_custom_vjp_matches_fd():
    _require_x64()
    import jax
    import jax.numpy as jnp

    from bimfx.jax_linalg import solve_linear

    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    A = A @ A.T + np.eye(4) * 0.5
    b = rng.normal(size=(4,))

    def loss_np(A_np: np.ndarray) -> float:
        x = np.linalg.solve(A_np, b)
        return float(np.sum(x**2))

    def loss_jax(A_j: jnp.ndarray) -> jnp.ndarray:
        x = solve_linear(A_j, jnp.asarray(b))
        return jnp.sum(x**2)

    grad_jax = np.asarray(jax.grad(loss_jax)(jnp.asarray(A)))
    grad_fd = _finite_diff_grad(loss_np, A)
    rel_err = np.linalg.norm(grad_jax - grad_fd) / np.maximum(1e-12, np.linalg.norm(grad_fd))
    assert rel_err < 5e-3


def test_solve_spd_custom_vjp_matches_fd():
    _require_x64()
    import jax
    import jax.numpy as jnp

    from bimfx.jax_linalg import solve_spd

    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3))
    A = A @ A.T + np.eye(3) * 0.2
    b = rng.normal(size=(3,))

    def loss_np(A_np: np.ndarray) -> float:
        x = np.linalg.solve(A_np, b)
        return float(np.sum(x**2))

    def loss_jax(A_j: jnp.ndarray) -> jnp.ndarray:
        x = solve_spd(A_j, jnp.asarray(b))
        return jnp.sum(x**2)

    grad_jax = np.asarray(jax.grad(loss_jax)(jnp.asarray(A)))
    grad_fd = _finite_diff_grad(loss_np, A)
    rel_err = np.linalg.norm(grad_jax - grad_fd) / np.maximum(1e-12, np.linalg.norm(grad_fd))
    assert rel_err < 5e-3
