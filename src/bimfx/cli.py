from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from bimfx import solve_bim, solve_bim_jax, solve_mfs, solve_mfs_jax
from bimfx.io import load_boundary
from bimfx.validation import relative_boundary_residual, summary_stats


def main() -> None:
    p = argparse.ArgumentParser(description="BIMFx CLI: solve/validate from an input boundary.")
    p.add_argument("--input", required=True, help="Boundary file (CSV/NC/NPZ/NPY/mesh).")
    p.add_argument("--normals", default=None, help="Normals CSV (for CSV input).")
    p.add_argument("--format", default=None, help="Override input format.")
    p.add_argument("--estimate-normals", action="store_true", help="Estimate normals for raw point clouds.")
    p.add_argument("--normal-k", type=int, default=20)
    p.add_argument("--n-points", type=int, default=2048, help="Mesh sampling points (for mesh inputs).")
    p.add_argument("--even", action="store_true", help="Even mesh sampling.")
    p.add_argument("--method", choices=["mfs", "bim", "mfs-jax", "bim-jax"], default="mfs")
    p.add_argument("--toroidal-flux", type=float, default=None)
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--outdir", default="outputs/cli")
    args = p.parse_args()

    data = load_boundary(
        args.input,
        normals_path=args.normals,
        format=args.format,
        estimate_normals=args.estimate_normals,
        normal_k=args.normal_k,
        n_points=args.n_points,
        even=args.even,
    )
    P = data.points
    N = data.normals
    if N is None:
        raise ValueError("Normals are required for solve. Provide normals or use --estimate-normals.")
    if args.subsample and args.subsample < len(P):
        idx = np.linspace(0, len(P) - 1, args.subsample, dtype=int)
        P = P[idx]
        N = N[idx]

    harmonic_coeffs = None
    if args.toroidal_flux is not None:
        harmonic_coeffs = (float(args.toroidal_flux) / (2.0 * np.pi), 0.0)

    if args.method == "mfs":
        field = solve_mfs(P, N, harmonic_coeffs=harmonic_coeffs)
    elif args.method == "bim":
        field = solve_bim(P, N, harmonic_coeffs=harmonic_coeffs)
    elif args.method == "mfs-jax":
        field = solve_mfs_jax(P, N, harmonic_coeffs=harmonic_coeffs)
    else:
        field = solve_bim_jax(P, N, harmonic_coeffs=harmonic_coeffs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.validate:
        res = relative_boundary_residual(field.B, P, N)
        stats = summary_stats(res)
        with (outdir / "summary.json").open("w") as f:
            json.dump(stats, f, indent=2)
        print(stats)
    else:
        print(f"[OK] Solve complete: method={args.method}, points={len(P)}")


if __name__ == "__main__":
    main()
