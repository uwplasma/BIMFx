from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bimfx import solve_bim, solve_bim_jax, solve_mfs, solve_mfs_jax
from bimfx.io import load_boundary
from bimfx.validation import relative_boundary_residual, summary_stats
from bimfx.vacuum.solve import SolveOptions


@dataclass(frozen=True)
class PipelineResult:
    stats: dict[str, Any]
    metadata: dict[str, Any]


__all__ = ["PipelineResult", "run_pipeline"]


def _load_config(path: str | Path) -> dict[str, Any]:
    try:
        import tomllib
    except Exception as exc:  # pragma: no cover
        raise ImportError("Python 3.11+ is required for TOML configs.") from exc
    with open(path, "rb") as f:
        return tomllib.load(f)


def run_pipeline(config_path: str | Path) -> PipelineResult:
    cfg = _load_config(config_path)
    boundary_cfg = cfg.get("boundary", {})
    solve_cfg = cfg.get("solve", {})
    validate_cfg = cfg.get("validate", {})
    output_cfg = cfg.get("output", {})

    data = load_boundary(
        boundary_cfg["input"],
        normals_path=boundary_cfg.get("normals"),
        format=boundary_cfg.get("format"),
        estimate_normals=bool(boundary_cfg.get("estimate_normals", False)),
        normal_k=int(boundary_cfg.get("normal_k", 20)),
        n_points=int(boundary_cfg.get("n_points", 2048)),
        even=bool(boundary_cfg.get("even", False)),
    )

    P = data.points
    N = data.normals
    if N is None:
        raise ValueError("Normals required for pipeline; provide normals or enable estimate_normals.")

    subsample = int(boundary_cfg.get("subsample", 0))
    if subsample and subsample < len(P):
        idx = np.linspace(0, len(P) - 1, subsample, dtype=int)
        P = P[idx]
        N = N[idx]

    method = solve_cfg.get("method", "mfs")
    toroidal_flux = solve_cfg.get("toroidal_flux")
    harmonic_coeffs = None
    if toroidal_flux is not None:
        harmonic_coeffs = (float(toroidal_flux) / (2.0 * np.pi), 0.0)

    if method in {"mfs", "bim"}:
        options = SolveOptions(
            method="mfs" if method == "mfs" else "bim",
            k_nn=int(solve_cfg.get("k_nn", 48)),
            lambda_reg=float(solve_cfg.get("lambda_reg", 1e-6)),
            source_factor=float(solve_cfg.get("source_factor", 2.0)),
            clip_factor=float(solve_cfg.get("clip_factor", 0.2)),
            acceleration=str(solve_cfg.get("acceleration", "none")),
            accel_theta=float(solve_cfg.get("accel_theta", 0.6)),
            accel_leaf_size=int(solve_cfg.get("accel_leaf_size", 64)),
            verbose=bool(solve_cfg.get("verbose", False)),
        )
        if method == "mfs":
            field = solve_mfs(P, N, harmonic_coeffs=harmonic_coeffs, options=options)
        else:
            field = solve_bim(P, N, harmonic_coeffs=harmonic_coeffs, options=options)
    elif method == "mfs-jax":
        field = solve_mfs_jax(
            P,
            N,
            k_nn=int(solve_cfg.get("k_nn", 48)),
            lambda_reg=float(solve_cfg.get("lambda_reg", 1e-6)),
            harmonic_coeffs=harmonic_coeffs,
        )
    elif method == "bim-jax":
        field = solve_bim_jax(
            P,
            N,
            k_nn=int(solve_cfg.get("k_nn", 48)),
            lambda_reg=float(solve_cfg.get("lambda_reg", 1e-6)),
            clip_factor=float(solve_cfg.get("clip_factor", 0.2)),
            harmonic_coeffs=harmonic_coeffs,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    eps_factor = float(validate_cfg.get("eps_factor", 0.02))
    res = relative_boundary_residual(field.B, P, N, eps_factor=eps_factor)
    stats = summary_stats(res)

    outdir = Path(output_cfg.get("dir", "outputs/pipeline"))
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.json").open("w") as f:
        json.dump(stats, f, indent=2)

    metadata = {
        "method": method,
        "input": str(boundary_cfg.get("input")),
        "subsample": subsample,
        "toroidal_flux": toroidal_flux,
    }
    return PipelineResult(stats=stats, metadata=metadata)
