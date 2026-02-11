#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from bimfx import solve_bim, solve_mfs
from bimfx.io import boundary_from_vmec_wout, load_boundary_csv
from bimfx.validation import relative_boundary_residual, summary_stats
from bimfx.vacuum.solve import SolveOptions


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
            "figure.figsize": (7.0, 4.4),
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


def _plot_boundary_grid(plt, datasets, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.0))
    axes = axes.ravel()
    for ax, (name, P) in zip(axes, datasets):
        sc = ax.scatter(P[:, 0], P[:, 1], c=P[:, 2], s=4, cmap="viridis", alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", "box")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle("Boundary point clouds (XY projection, colored by z)")
    fig.savefig(outpath)
    plt.close(fig)


def _plot_bars(plt, labels, mfs_vals, bim_vals, outpath: Path, ylabel: str, title: str) -> None:
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, mfs_vals, width, label="MFS")
    ax.bar(x + width / 2, bim_vals, width, label="BIM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(outpath)
    plt.close(fig)


def _benchmark_dataset(P: np.ndarray, N: np.ndarray, toroidal_flux: float | None, options: SolveOptions) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for method in ("mfs", "bim"):
        t0 = perf_counter()
        if method == "mfs":
            field = solve_mfs(P, N, toroidal_flux=toroidal_flux, options=options)
        else:
            field = solve_bim(P, N, toroidal_flux=toroidal_flux, options=options)
        dt = perf_counter() - t0
        res = relative_boundary_residual(field.B, P, N, eps_factor=0.02)
        stats = summary_stats(res)
        stats["time_sec"] = dt
        results[method] = stats
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Generate benchmark gallery figures and tables.")
    p.add_argument("--outdir", default="docs/_static/benchmark")
    p.add_argument("--subsample", type=int, default=800)
    args = p.parse_args()

    plt = _maybe_matplotlib()
    if plt is None:
        raise SystemExit("matplotlib is required for benchmark figure generation.")
    _apply_style(plt)

    repo_root = Path(__file__).resolve().parents[1]
    outdir = repo_root / args.outdir
    _ensure_outdir(outdir)

    inputs = repo_root / "inputs"

    datasets = [
        ("VMEC QA", boundary_from_vmec_wout(inputs / "wout_precise_QA.nc", s=1.0, ntheta=40, nphi=80)),
        ("Near-axis", load_boundary_csv(inputs / "wout_LandremanSenguptaPlunk_5.3.csv", inputs / "wout_LandremanSenguptaPlunk_5.3_normals.csv")),
        ("Mirror", load_boundary_csv(inputs / "sflm_rm4.csv", inputs / "sflm_rm4_normals.csv")),
        ("Knot", load_boundary_csv(inputs / "knot_tube.csv", inputs / "knot_tube_normals.csv")),
    ]

    grid_data = []
    for name, data in datasets:
        P, _ = _subsample(data.points, data.normals, args.subsample)
        grid_data.append((name, P))
    _plot_boundary_grid(plt, grid_data, outdir / "boundary_grid.png")

    options = SolveOptions(k_nn=24, lambda_reg=1e-6, verbose=False)
    rows = []
    for name, data in datasets:
        P, N = _subsample(data.points, data.normals, min(args.subsample, 600))
        toroidal_flux = 1.0 if name in {"VMEC QA", "Near-axis"} else None
        results = _benchmark_dataset(P, N, toroidal_flux, options)
        rows.append((name, results["mfs"], results["bim"]))

    labels = [r[0] for r in rows]
    mfs_time = [r[1]["time_sec"] for r in rows]
    bim_time = [r[2]["time_sec"] for r in rows]
    mfs_rms = [r[1]["rms"] for r in rows]
    bim_rms = [r[2]["rms"] for r in rows]

    _plot_bars(plt, labels, mfs_time, bim_time, outdir / "benchmark_time.png", "time (s)", "Solve time")
    _plot_bars(plt, labels, mfs_rms, bim_rms, outdir / "benchmark_rms.png", "RMS(n·B)/median(|B|)", "Boundary residual")

    table_md = [
        "| dataset | method | rms | p95 | max | time_sec |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, mfs_stats, bim_stats in rows:
        for method, stats in (("mfs", mfs_stats), ("bim", bim_stats)):
            table_md.append(
                f"| {name} | {method} | {stats['rms']:.3e} | {stats['p95']:.3e} | {stats['max']:.3e} | {stats['time_sec']:.3f} |"
            )

    (outdir / "benchmarks.md").write_text("\n".join(table_md) + "\n")
    (outdir / "benchmarks.csv").write_text(
        "dataset,method,rms,p95,max,time_sec\n"
        + "\n".join(
            f"{name},{method},{stats['rms']:.6e},{stats['p95']:.6e},{stats['max']:.6e},{stats['time_sec']:.6f}"
            for name, mfs_stats, bim_stats in rows
            for method, stats in (("mfs", mfs_stats), ("bim", bim_stats))
        )
        + "\n"
    )

    print(f"[OK] Wrote benchmark images and tables to {outdir}")


if __name__ == "__main__":
    main()
