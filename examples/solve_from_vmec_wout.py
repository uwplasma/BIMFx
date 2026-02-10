#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from bimfx import solve_bim, solve_mfs
from bimfx.io import boundary_from_vmec_wout
from bimfx.validation import boundary_normal_residual


def main() -> None:
    p = argparse.ArgumentParser(description="Solve a vacuum field from a VMEC wout file.")
    p.add_argument("--wout", required=True, help="Path to wout*.nc")
    p.add_argument("--method", choices=["mfs", "bim"], default="mfs")
    p.add_argument("--ntheta", type=int, default=48)
    p.add_argument("--nphi", type=int, default=96)
    p.add_argument("--toroidal-flux", type=float, default=1.0)
    p.add_argument("--subsample", type=int, default=300)
    args = p.parse_args()

    data = boundary_from_vmec_wout(args.wout, s=1.0, ntheta=args.ntheta, nphi=args.nphi)
    P, N = data.points, data.normals
    if args.subsample and args.subsample < len(P):
        idx = np.linspace(0, len(P) - 1, args.subsample, dtype=int)
        P = P[idx]
        N = N[idx]

    if args.method == "mfs":
        field = solve_mfs(P, N, toroidal_flux=args.toroidal_flux)
    else:
        field = solve_bim(P, N, toroidal_flux=args.toroidal_flux)

    Pin = P - 0.02 * np.median(np.linalg.norm(P - P.mean(axis=0), axis=1)) * N
    res = boundary_normal_residual(field.B, Pin, N, normalize=True)
    print(f"method={args.method}  rms(n·B/|B|)={np.sqrt(np.mean(res**2)):.3e}")


if __name__ == "__main__":
    main()
