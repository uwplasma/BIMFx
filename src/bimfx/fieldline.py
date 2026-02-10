from __future__ import annotations

from typing import Callable

import numpy as np

from bimfx.tracing import trace_fieldlines_rk4


def trace_fieldline_rk4(
    B: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    *,
    ds: float,
    n_steps: int,
    normalize: bool = True,
) -> np.ndarray:
    """Trace a single field line with a fixed-step RK4 integrator."""
    trace = trace_fieldlines_rk4(B, np.asarray(x0)[None, :], ds=ds, n_steps=n_steps, normalize=normalize)
    return trace.trajectories[0]
