from __future__ import annotations

from typing import Any
from inspect import Signature, Parameter

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

Array = Any

__all__ = ["solve_linear", "solve_spd"]


@jax.custom_vjp
def solve_linear(A: Array, b: Array) -> Array:
    """Solve a linear system with implicit differentiation.

    This uses a custom VJP so gradients do not backpropagate through the
    factorization itself, improving stability for large systems.
    """
    return jnp.linalg.solve(A, b)


def _solve_linear_fwd(A: Array, b: Array) -> tuple[Array, tuple[Array, Array]]:
    x = jnp.linalg.solve(A, b)
    return x, (A, x)


def _solve_linear_bwd(res: tuple[Array, Array], g: Array) -> tuple[Array, Array]:
    A, x = res
    y = jnp.linalg.solve(A.T, g)
    if g.ndim == 1:
        grad_A = -jnp.outer(y, x)
        grad_b = y
    else:
        grad_A = -y @ x.T
        grad_b = y
    return grad_A, grad_b


solve_linear.defvjp(_solve_linear_fwd, _solve_linear_bwd)


@jax.custom_vjp
def solve_spd(A: Array, b: Array) -> Array:
    """Solve a symmetric positive definite system with implicit differentiation."""
    L = jnp.linalg.cholesky(A)
    y = jsp_linalg.solve_triangular(L, b, lower=True)
    return jsp_linalg.solve_triangular(L.T, y, lower=False)


def _solve_spd_fwd(A: Array, b: Array) -> tuple[Array, tuple[Array, Array]]:
    L = jnp.linalg.cholesky(A)
    y = jsp_linalg.solve_triangular(L, b, lower=True)
    x = jsp_linalg.solve_triangular(L.T, y, lower=False)
    return x, (A, x)


def _solve_spd_bwd(res: tuple[Array, Array], g: Array) -> tuple[Array, Array]:
    A, x = res
    y = jnp.linalg.solve(A, g)
    if g.ndim == 1:
        grad_A = -jnp.outer(y, x)
        grad_b = y
    else:
        grad_A = -y @ x.T
        grad_b = y
    return grad_A, grad_b


solve_spd.defvjp(_solve_spd_fwd, _solve_spd_bwd)

_sig = Signature(
    parameters=[
        Parameter("A", Parameter.POSITIONAL_OR_KEYWORD),
        Parameter("b", Parameter.POSITIONAL_OR_KEYWORD),
    ]
)
solve_linear.__signature__ = _sig
solve_spd.__signature__ = _sig
