from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from bimfx.tracing import trace_fieldlines_rk4


def sample_psi_along_fieldlines(
    psi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    B: Callable[[Any], Any],
    seeds: np.ndarray,
    *,
    ds: float,
    n_steps: int,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace field lines and sample psi along them.

    Returns (s, psi_samples) where:
      - s is 1D arclength-like coordinate
      - psi_samples is (n_seed, n_steps+1)
    """
    trace = trace_fieldlines_rk4(B, seeds, ds=ds, n_steps=n_steps, normalize=normalize)
    traj = trace.trajectories
    interp = RegularGridInterpolator((xs, ys, zs), psi, bounds_error=False, fill_value=np.nan)

    n_seed, n_step, _ = traj.shape
    pts = traj.reshape(-1, 3)
    vals = interp(pts).reshape(n_seed, n_step)
    s = np.arange(n_step) * ds
    return s, vals

