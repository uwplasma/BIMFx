#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bimfx import solve_bim, solve_mfs


def main() -> None:
    p = argparse.ArgumentParser(description="Solve a vacuum field from a boundary point cloud + normals (CSV).")
    p.add_argument("--method", choices=["mfs", "bim"], default="mfs")
    p.add_argument("--xyz", required=True, help="CSV with columns x,y,z (header row allowed).")
    p.add_argument("--normals", required=True, help="CSV with columns nx,ny,nz (header row allowed).")
    p.add_argument("--toroidal-flux", type=float, default=1.0, help="Toroidal flux Φ_t used to set a_t=Φ_t/(2π).")
    p.add_argument("--subsample", type=int, default=0, help="If >0, uniformly subsample to this many points.")
    p.add_argument("--eps-factor", type=float, default=5e-3, help="Interior offset as eps_factor * median neighbor scale.")
    args = p.parse_args()

    P = np.loadtxt(args.xyz, delimiter=",", skiprows=1)
    N = np.loadtxt(args.normals, delimiter=",", skiprows=1)
    if args.subsample and args.subsample < len(P):
        idx = np.linspace(0, len(P) - 1, args.subsample, dtype=int)
        P = P[idx]
        N = N[idx]

    Nn = N / np.maximum(1e-30, np.linalg.norm(N, axis=1, keepdims=True))
    if args.method == "mfs":
        field = solve_mfs(P, Nn, toroidal_flux=args.toroidal_flux)
    else:
        field = solve_bim(P, Nn, toroidal_flux=args.toroidal_flux)

    # Check the boundary condition slightly inside (avoid singular evaluation at Γ).
    P_in = P - args.eps_factor * np.median(np.linalg.norm(P - P.mean(axis=0), axis=1)) * Nn
    B_in = np.asarray(field.B(P_in))
    ndot = np.sum(Nn * B_in, axis=1)
    print(f"method={args.method}  rms(n·B)={np.sqrt(np.mean(ndot**2)):.3e}  max|n·B|={np.max(np.abs(ndot)):.3e}")


if __name__ == "__main__":
    main()

