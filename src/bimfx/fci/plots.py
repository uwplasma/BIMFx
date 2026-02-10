from __future__ import annotations

from typing import Iterable

import numpy as np

from bimfx.fci.workflows import PsiSlice
from bimfx.tracing import PoincareSection


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting.") from exc
    return plt


def plot_psi_along_fieldlines(
    s: np.ndarray,
    psi_along: np.ndarray,
    *,
    ax=None,
    alpha: float = 0.7,
):
    """Plot psi(s) along field lines."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for k in range(psi_along.shape[0]):
        ax.plot(s, psi_along[k, :], lw=1.0, alpha=alpha)
    ax.set_xlabel("s")
    ax.set_ylabel("psi")
    ax.set_title("Psi along field lines")
    return ax


def plot_poincare_with_psi_contours(
    poincare: PoincareSection,
    psi_slice: PsiSlice,
    *,
    ax=None,
    levels: int | Iterable[float] = 20,
    s: float = 6.0,
):
    """Plot Poincare points with psi(R,Z) contours at a given phi plane."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2))

    RR, ZZ = np.meshgrid(psi_slice.R, psi_slice.Z, indexing="xy")
    ax.contour(RR, ZZ, psi_slice.psi, levels=levels, cmap="viridis", linewidths=0.6)

    ax.scatter(poincare.R, poincare.Z, s=s, c="k", alpha=0.6)
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(f"Poincare + psi contours (phi={psi_slice.phi:.3f})")
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_poincare_overlays(
    poincare: PoincareSection,
    psi_slices: list[PsiSlice],
    *,
    levels: int | Iterable[float] = 20,
    s: float = 6.0,
):
    """Plot one panel per phi plane using matching slices."""
    plt = _require_matplotlib()
    n = max(1, len(psi_slices))
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2), squeeze=False)

    for k, slc in enumerate(psi_slices):
        ax = axes[0, k]
        mask = poincare.plane_index == k
        pc = PoincareSection(
            R=poincare.R[mask],
            Z=poincare.Z[mask],
            seed_index=poincare.seed_index[mask],
            plane_index=poincare.plane_index[mask],
            phi_planes=poincare.phi_planes,
        )
        plot_poincare_with_psi_contours(pc, slc, ax=ax, levels=levels, s=s)

    return fig, axes
