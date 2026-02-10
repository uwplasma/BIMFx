#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from bimfx import poincare_sections, trace_fieldlines_rk4


def B(points: np.ndarray) -> np.ndarray:
    # Rotation around z-axis: circles in xy-plane
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.stack([-y, x, 0.0 * z], axis=1)


def main() -> None:
    seeds = np.array([[1.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.8, 0.0, 0.0]])
    trace = trace_fieldlines_rk4(B, seeds, ds=0.05, n_steps=500, normalize=True)
    sec = poincare_sections(trace.trajectories, phi_planes=[0.0], nfp=1)
    print(f"Computed {sec.R.size} Poincare points")


if __name__ == "__main__":
    main()

