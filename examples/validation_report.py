#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from bimfx import solve_bim, solve_mfs
from bimfx.io import boundary_from_vmec_wout, load_boundary_csv
from bimfx.validation import relative_boundary_residual, summary_stats
from bimfx.vacuum.solve import SolveOptions

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    kind: str
    loader: Callable[[], tuple[np.ndarray, np.ndarray]]


def _subsample(P: np.ndarray, N: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0 or P.shape[0] <= n:
        return P, N
    idx = np.linspace(0, P.shape[0] - 1, n, dtype=int)
    return P[idx], N[idx]


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _maybe_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    return plt


def _apply_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 4.0),
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


def _solve_field(
    method: str,
    P: np.ndarray,
    N: np.ndarray,
    *,
    toroidal_flux: float | None,
    options: SolveOptions,
):
    if method == "mfs":
        return solve_mfs(P, N, toroidal_flux=toroidal_flux, options=options)
    if method == "bim":
        return solve_bim(P, N, toroidal_flux=toroidal_flux, options=options)
    raise ValueError(f"Unknown method: {method}")


def _evaluate(
    method: str,
    P: np.ndarray,
    N: np.ndarray,
    *,
    toroidal_flux: float | None,
    options: SolveOptions,
    eps_factor: float,
) -> dict[str, float]:
    t0 = perf_counter()
    field = _solve_field(method, P, N, toroidal_flux=toroidal_flux, options=options)
    res = relative_boundary_residual(field.B, P, N, eps_factor=eps_factor)
    stats = summary_stats(res)
    stats["time_sec"] = perf_counter() - t0
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Generate validation sweep report for BIMFx.")
    p.add_argument("--outdir", default="outputs/validation_report")
    p.add_argument("--subsample-base", type=int, default=400)
    p.add_argument("--eps-factor", type=float, default=0.02)
    p.add_argument("--k-nn", type=int, nargs="+", default=[12, 24, 36, 48, 64, 96])
    p.add_argument("--subsample", type=int, nargs="+", default=[150, 300, 600, 900])
    p.add_argument("--lambda-reg", type=float, nargs="+", default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    p.add_argument("--ci", action="store_true", help="Use a small CI-friendly sweep.")
    p.add_argument("--datasets", nargs="*", default=None, help="Subset of datasets by name.")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    inputs = repo_root / "inputs"
    outdir = repo_root / args.outdir
    _ensure_outdir(outdir)

    def _load_csv(stem: str) -> tuple[np.ndarray, np.ndarray]:
        data = load_boundary_csv(inputs / f"{stem}.csv", inputs / f"{stem}_normals.csv")
        return data.points, data.normals

    def _load_wout(name: str, ntheta: int, nphi: int) -> tuple[np.ndarray, np.ndarray]:
        data = boundary_from_vmec_wout(inputs / name, s=1.0, ntheta=ntheta, nphi=nphi)
        return data.points, data.normals

    datasets = [
        DatasetConfig(
            name="wout_precise_QA",
            kind="torus",
            loader=lambda: _load_wout("wout_precise_QA.nc", ntheta=48, nphi=96),
        ),
        DatasetConfig(
            name="wout_SLAM_6_coils_wout",
            kind="torus",
            loader=lambda: _load_wout("wout_SLAM_6_coils.nc", ntheta=32, nphi=64),
        ),
        DatasetConfig(
            name="wout_SLAM_6_coils",
            kind="torus",
            loader=lambda: _load_csv("wout_SLAM_6_coils"),
        ),
        DatasetConfig(
            name="wout_LandremanSenguptaPlunk_5.3",
            kind="torus",
            loader=lambda: _load_csv("wout_LandremanSenguptaPlunk_5.3"),
        ),
        DatasetConfig(
            name="sflm_rm4",
            kind="mirror",
            loader=lambda: _load_csv("sflm_rm4"),
        ),
        DatasetConfig(
            name="knot_tube",
            kind="other",
            loader=lambda: _load_csv("knot_tube"),
        ),
    ]

    if args.ci or os.environ.get("CI"):
        args.subsample_base = min(args.subsample_base, 300)
        args.k_nn = [12, 24, 48]
        args.subsample = [150, 300]
        args.lambda_reg = [1e-6, 1e-4]
        datasets = [d for d in datasets if d.name in {"wout_precise_QA", "wout_SLAM_6_coils"}]
        datasets = [
            DatasetConfig(
                name="wout_precise_QA",
                kind="torus",
                loader=lambda: _load_wout("wout_precise_QA.nc", ntheta=24, nphi=48),
            ),
            DatasetConfig(
                name="wout_SLAM_6_coils",
                kind="torus",
                loader=lambda: _load_csv("wout_SLAM_6_coils"),
            ),
        ]

    if args.datasets:
        wanted = set(args.datasets)
        datasets = [d for d in datasets if d.name in wanted]

    summary_csv = outdir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "method",
                "k_nn",
                "lambda_reg",
                "subsample",
                "rms",
                "p95",
                "max",
                "bim_mfs_diff_rms",
                "time_sec",
            ]
        )

        for ds in datasets:
            P, N = ds.loader()
            P, N = _subsample(P, N, args.subsample_base)
            toroidal_flux = 1.0 if ds.kind == "torus" else None

            # Baseline MFS/BIM
            base_opts = SolveOptions(k_nn=24, lambda_reg=1e-6, verbose=False)
            base_fields = {}
            for method in ("mfs", "bim"):
                stats = _evaluate(
                    method,
                    P,
                    N,
                    toroidal_flux=toroidal_flux,
                    options=base_opts,
                    eps_factor=args.eps_factor,
                )
                base_fields[method] = stats

            # Cross-validation on probe set
            probe = P - args.eps_factor * np.median(np.linalg.norm(P - P.mean(axis=0), axis=1)) * N
            field_mfs = _solve_field("mfs", P, N, toroidal_flux=toroidal_flux, options=base_opts)
            field_bim = _solve_field("bim", P, N, toroidal_flux=toroidal_flux, options=base_opts)
            Bm = np.asarray(field_mfs.B(probe))
            Bb = np.asarray(field_bim.B(probe))
            diff = np.linalg.norm(Bm - Bb, axis=1)
            diff_rms = float(np.sqrt(np.mean(diff**2)))

            for method in ("mfs", "bim"):
                stats = base_fields[method]
                writer.writerow(
                    [
                        ds.name,
                        method,
                        base_opts.k_nn,
                        base_opts.lambda_reg,
                        args.subsample_base,
                        stats["rms"],
                        stats["p95"],
                        stats["max"],
                        diff_rms,
                        stats["time_sec"],
                    ]
                )

            # BIM sweeps
            for k_nn in args.k_nn:
                if k_nn >= P.shape[0]:
                    continue
                opts = SolveOptions(k_nn=int(k_nn), lambda_reg=1e-6, verbose=False)
                stats = _evaluate(
                    "bim",
                    P,
                    N,
                    toroidal_flux=toroidal_flux,
                    options=opts,
                    eps_factor=args.eps_factor,
                )
                writer.writerow(
                    [
                        ds.name,
                        "bim_k_nn",
                        opts.k_nn,
                        opts.lambda_reg,
                        args.subsample_base,
                        stats["rms"],
                        stats["p95"],
                        stats["max"],
                        "",
                        stats["time_sec"],
                    ]
                )

            for n in args.subsample:
                Pn, Nn = _subsample(P, N, int(n))
                if Pn.shape[0] < 50:
                    continue
                opts = SolveOptions(k_nn=min(24, Pn.shape[0] - 1), lambda_reg=1e-6, verbose=False)
                stats = _evaluate(
                    "bim",
                    Pn,
                    Nn,
                    toroidal_flux=toroidal_flux,
                    options=opts,
                    eps_factor=args.eps_factor,
                )
                writer.writerow(
                    [
                        ds.name,
                        "bim_subsample",
                        opts.k_nn,
                        opts.lambda_reg,
                        int(n),
                        stats["rms"],
                        stats["p95"],
                        stats["max"],
                        "",
                        stats["time_sec"],
                    ]
                )

            for lam in args.lambda_reg:
                opts = SolveOptions(k_nn=24, lambda_reg=float(lam), verbose=False)
                stats = _evaluate(
                    "bim",
                    P,
                    N,
                    toroidal_flux=toroidal_flux,
                    options=opts,
                    eps_factor=args.eps_factor,
                )
                writer.writerow(
                    [
                        ds.name,
                        "bim_lambda_reg",
                        opts.k_nn,
                        opts.lambda_reg,
                        args.subsample_base,
                        stats["rms"],
                        stats["p95"],
                        stats["max"],
                        "",
                        stats["time_sec"],
                    ]
                )

    plt = _maybe_matplotlib()
    if plt is None:
        print("[WARN] matplotlib not available; skipping plots.")
        return

    _apply_plot_style(plt)

    data = np.genfromtxt(summary_csv, delimiter=",", names=True, dtype=None, encoding=None)

    def plot_sweep(filter_method: str, xfield: str, xlabel: str, outfile: Path, logx: bool = False) -> None:
        fig, ax = plt.subplots()
        for ds in datasets:
            mask = (data["dataset"] == ds.name) & (data["method"] == filter_method)
            xs = data[xfield][mask]
            ys = data["rms"][mask]
            if xs.size == 0:
                continue
            order = np.argsort(xs)
            ax.plot(xs[order], ys[order], marker="o", label=ds.name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RMS(|n·B|)/median(|B|)")
        ax.set_title(filter_method)
        if logx:
            ax.set_xscale("log")
        ax.legend()
        fig.savefig(outfile)
        plt.close(fig)

    plot_sweep("bim_k_nn", "k_nn", "k_nn", outdir / "sweep_k_nn.png")
    plot_sweep("bim_subsample", "subsample", "subsample", outdir / "sweep_subsample.png")
    plot_sweep("bim_lambda_reg", "lambda_reg", "lambda_reg", outdir / "sweep_lambda_reg.png", logx=True)

    summary_md = outdir / "summary.md"
    with summary_md.open("w") as f:
        f.write("# Validation summary\n\n")
        f.write("Base metrics (k_nn=24, lambda_reg=1e-6, subsample=base):\n\n")
        f.write("| dataset | method | rms | p95 | max | bim_mfs_diff_rms | time_sec |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for ds in datasets:
            for method in ("mfs", "bim"):
                mask = (data["dataset"] == ds.name) & (data["method"] == method)
                if not np.any(mask):
                    continue
                row = data[mask][0]
                f.write(
                    f"| {row['dataset']} | {row['method']} | {row['rms']:.3e} | {row['p95']:.3e} | {row['max']:.3e} | {row['bim_mfs_diff_rms']} | {row['time_sec']:.2f} |\n"
                )

    print(f"[OK] Wrote {summary_csv}")
    print(f"[OK] Wrote {summary_md}")
    print(f"[OK] Plots in {outdir}")


if __name__ == "__main__":
    main()
