#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from bimfx.io import (
    boundary_from_sflm_npy,
    boundary_from_slam_npz,
    boundary_from_stl,
    boundary_from_vmec_wout,
    save_boundary_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert boundary sources to BIMFx CSV format.")
    ap.add_argument("--mode", choices=["wout", "slam", "sflm", "stl"], required=True)
    ap.add_argument("--in", dest="inp", required=True, help="Input file path")
    ap.add_argument("--out-stem", required=True, help="Output stem for CSVs (no extension)")
    ap.add_argument("--s", type=float, default=1.0, help="VMEC radial location (wout only)")
    ap.add_argument("--ntheta", type=int, default=64, help="VMEC/SLAM poloidal resolution")
    ap.add_argument("--nphi", type=int, default=128, help="VMEC/SLAM toroidal resolution")
    ap.add_argument("--npoints", type=int, default=2048, help="STL sampling points")
    ap.add_argument("--even", action="store_true", help="STL even-ish sampling (trimesh)")
    args = ap.parse_args()

    if args.mode == "wout":
        data = boundary_from_vmec_wout(args.inp, s=args.s, ntheta=args.ntheta, nphi=args.nphi)
    elif args.mode == "slam":
        data = boundary_from_slam_npz(args.inp, ntheta=args.ntheta, nphi=args.nphi)
    elif args.mode == "sflm":
        data = boundary_from_sflm_npy(args.inp)
    else:
        data = boundary_from_stl(args.inp, n_points=args.npoints, even=args.even)

    out_stem = Path(args.out_stem)
    pts_path, nrm_path = save_boundary_csv(data.points, data.normals, out_stem)
    print(f"[SAVE] points -> {pts_path}")
    if nrm_path:
        print(f"[SAVE] normals -> {nrm_path}")


if __name__ == "__main__":
    main()

