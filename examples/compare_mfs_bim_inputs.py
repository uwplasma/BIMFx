#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np

from bimfx import solve_bim, solve_mfs
from bimfx.io import boundary_from_vmec_wout, load_boundary_csv
from bimfx.validation import boundary_normal_residual, summary_stats


def _subsample(P: np.ndarray, N: np.ndarray, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    if P.shape[0] <= n:
        return P, N
    idx = np.linspace(0, P.shape[0] - 1, n, dtype=int)
    return P[idx], N[idx]


def _offset_inside(P: np.ndarray, N: np.ndarray, eps_factor: float = 0.02) -> np.ndarray:
    center = np.mean(P, axis=0)
    scale = np.median(np.linalg.norm(P - center[None, :], axis=1))
    return P - eps_factor * scale * N


def _report(label: str, res: np.ndarray) -> None:
    stats = summary_stats(res)
    print(f"{label}: rms={stats['rms']:.3e}  p95={stats['p95']:.3e}  max={stats['max']:.3e}")


def main() -> None:
    inputs = Path(__file__).resolve().parents[1] / "inputs"

    csv_cases = [
        ("knot", "other", inputs / "knot_tube.csv", inputs / "knot_tube_normals.csv"),
        ("mirror", "mirror", inputs / "sflm_rm4.csv", inputs / "sflm_rm4_normals.csv"),
        ("near_axis", "torus", inputs / "wout_LandremanSenguptaPlunk_5.3.csv", inputs / "wout_LandremanSenguptaPlunk_5.3_normals.csv"),
        ("vmec_csv", "torus", inputs / "wout_SLAM_6_coils.csv", inputs / "wout_SLAM_6_coils_normals.csv"),
    ]

    print("=== CSV inputs ===")
    for name, kind, xyz, nrm in csv_cases:
        data = load_boundary_csv(xyz, nrm)
        P, N = _subsample(data.points, data.normals)
        Pin = _offset_inside(P, N)

        toroidal_flux = 1.0 if kind == "torus" else None
        field_mfs = solve_mfs(P, N, toroidal_flux=toroidal_flux)
        field_bim = solve_bim(P, N, toroidal_flux=toroidal_flux)

        res_mfs = boundary_normal_residual(field_mfs.B, Pin, N, normalize=True)
        res_bim = boundary_normal_residual(field_bim.B, Pin, N, normalize=True)

        _report(f"{name} / mfs", res_mfs)
        _report(f"{name} / bim", res_bim)

    print("\n=== VMEC wout ===")
    wout = inputs / "wout_precise_QA.nc"
    data = boundary_from_vmec_wout(wout, s=1.0, ntheta=32, nphi=64)
    P, N = _subsample(data.points, data.normals)
    Pin = _offset_inside(P, N)

    field_mfs = solve_mfs(P, N, toroidal_flux=1.0)
    field_bim = solve_bim(P, N, toroidal_flux=1.0)

    res_mfs = boundary_normal_residual(field_mfs.B, Pin, N, normalize=True)
    res_bim = boundary_normal_residual(field_bim.B, Pin, N, normalize=True)
    _report("wout_precise_QA / mfs", res_mfs)
    _report("wout_precise_QA / bim", res_bim)


if __name__ == "__main__":
    main()
