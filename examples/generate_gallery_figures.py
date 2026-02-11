#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from bimfx import solve_bim, solve_mfs
from bimfx.io import boundary_from_vmec_wout, load_boundary_csv
from bimfx.tracing import poincare_sections, trace_fieldlines_rk4


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _maybe_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    return plt


def _apply_style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 4.2),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
        }
    )


def _subsample(P: np.ndarray, N: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray | None]:
    if n <= 0 or P.shape[0] <= n:
        return P, N
    idx = np.linspace(0, P.shape[0] - 1, n, dtype=int)
    if N is None:
        return P[idx], None
    return P[idx], N[idx]


def _torus_point_cloud(R: float, r: float, nphi: int, ntheta: int) -> tuple[np.ndarray, np.ndarray]:
    phis = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    points = []
    normals = []
    for phi in phis:
        cphi, sphi = np.cos(phi), np.sin(phi)
        for th in thetas:
            cth, sth = np.cos(th), np.sin(th)
            x = (R + r * cth) * cphi
            y = (R + r * cth) * sphi
            z = r * sth
            points.append([x, y, z])
            normals.append([cth * cphi, cth * sphi, sth])
    P = np.asarray(points, dtype=float)
    N = np.asarray(normals, dtype=float)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return P, N


def _plot_boundary(plt, P: np.ndarray, outpath: Path, title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=4, c=P[:, 2], cmap="viridis", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=35)
    fig.savefig(outpath)
    plt.close(fig)


def _plot_fieldlines(plt, traj: np.ndarray, outpath: Path, title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for k in range(min(10, traj.shape[0])):
        ax.plot(traj[k, :, 0], traj[k, :, 1], traj[k, :, 2], lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=40)
    fig.savefig(outpath)
    plt.close(fig)


def _plot_poincare(plt, poincare, outpath: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.scatter(poincare.R, poincare.Z, s=8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", "box")
    fig.savefig(outpath)
    plt.close(fig)


def _plot_bim_mfs_diff(plt, diff: np.ndarray, outpath: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.hist(diff, bins=40, color="#3b82f6", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("|B_mfs - B_bim|")
    ax.set_ylabel("count")
    fig.savefig(outpath)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate gallery figures for BIMFx docs/README.")
    p.add_argument("--outdir", default="docs/_static/gallery")
    p.add_argument("--subsample", type=int, default=1500)
    args = p.parse_args()

    plt = _maybe_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for figure generation.")
    _apply_style(plt)

    repo_root = Path(__file__).resolve().parents[1]
    outdir = repo_root / args.outdir
    _ensure_outdir(outdir)

    # Boundary plots (knot + VMEC)
    knot = load_boundary_csv(repo_root / "inputs/knot_tube.csv", repo_root / "inputs/knot_tube_normals.csv")
    Pk, _ = _subsample(knot.points, knot.normals, args.subsample)
    _plot_boundary(plt, Pk, outdir / "boundary_knot.png", "Knot boundary point cloud")

    wout = boundary_from_vmec_wout(repo_root / "inputs/wout_precise_QA.nc", s=1.0, ntheta=40, nphi=80)
    Pw, _ = _subsample(wout.points, wout.normals, args.subsample)
    _plot_boundary(plt, Pw, outdir / "boundary_vmec.png", "VMEC boundary point cloud")

    # Toroidal fieldlines + Poincare
    P, N = _torus_point_cloud(R=3.0, r=1.0, nphi=14, ntheta=14)
    field = solve_mfs(P, N, toroidal_flux=1.0)
    seeds = np.array(
        [
            [3.0 + 0.3 * np.cos(t), 0.0, 0.3 * np.sin(t)]
            for t in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
        ]
    )
    traces = trace_fieldlines_rk4(field.B, seeds, ds=0.05, n_steps=1200)
    _plot_fieldlines(plt, traces.trajectories, outdir / "fieldlines_torus.png", "Field-line traces (torus)")
    pcs = poincare_sections(traces.trajectories, phi_planes=[0.0])
    _plot_poincare(plt, pcs, outdir / "poincare_torus.png", "Poincaré section (torus)")

    # BIM vs MFS comparison on probe points
    field_bim = solve_bim(P, N, toroidal_flux=1.0)
    probe = P - 0.05 * N
    Bm = np.asarray(field.B(probe))
    Bb = np.asarray(field_bim.B(probe))
    diff = np.linalg.norm(Bm - Bb, axis=1)
    _plot_bim_mfs_diff(plt, diff, outdir / "bim_vs_mfs.png", "BIM vs MFS (probe set)")

    print(f"[OK] Gallery images written to {outdir}")


if __name__ == "__main__":
    main()
