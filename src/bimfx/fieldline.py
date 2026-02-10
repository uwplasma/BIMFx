from __future__ import annotations

from typing import Callable

import numpy as np


def trace_fieldline_rk4(
    B: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    ds: float,
    n_steps: int,
    normalize: bool = True,
) -> np.ndarray:
    """Trace a field line with a fixed-step RK4 integrator.

    Solves x'(s) = B(x) (optionally normalized to unit speed).

    Parameters
    ----------
    B:
        Callable returning an (n,3) array for input (n,3).
    x0:
        Initial condition, shape (3,).
    ds:
        Step size in arclength-like parameter.
    n_steps:
        Number of integration steps (trajectory has length n_steps+1).
    normalize:
        If True, integrate x'(s) = B / ||B|| to get roughly uniform step length.
    """
    x = np.asarray(x0, dtype=float).reshape(3)
    traj = np.empty((n_steps + 1, 3), dtype=float)
    traj[0] = x

    def f(xp: np.ndarray) -> np.ndarray:
        v = np.asarray(B(xp[None, :]))[0]
        if not normalize:
            return v
        nrm = np.linalg.norm(v)
        return v / max(1e-30, nrm)

    for k in range(n_steps):
        k1 = f(x)
        k2 = f(x + 0.5 * ds * k1)
        k3 = f(x + 0.5 * ds * k2)
        k4 = f(x + ds * k3)
        x = x + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[k + 1] = x
    return traj
