#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from bimfx.fci import solve_flux_psi_fci


def B(points: np.ndarray) -> np.ndarray:
    # Constant field along z
    return np.tile([0.0, 0.0, 1.0], (points.shape[0], 1))


def sphere_points(n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    P = rng.normal(size=(n, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    N = P.copy()
    return P, N


def main() -> None:
    P, N = sphere_points()
    sol = solve_flux_psi_fci(B, P, N, nx=32, ny=32, nz=32)
    print(f"psi stats: min={sol.psi.min():.3e}, max={sol.psi.max():.3e}")


if __name__ == "__main__":
    main()

