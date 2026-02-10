from __future__ import annotations

from .solve import FCISolution, solve_flux_psi_fci
from .diagnostics import field_alignment_error
from .analysis import sample_psi_along_fieldlines
from .plots import plot_poincare_overlays, plot_poincare_with_psi_contours, plot_psi_along_fieldlines
from .surfaces import extract_isosurfaces
from .workflows import (
    FluxSurface,
    FluxSurfaceAnalysis,
    PsiSlice,
    analyze_flux_surfaces,
    compute_psi_rz_slices,
    fit_flux_surfaces,
    seed_from_boundary,
)

__all__ = [
    "FCISolution",
    "solve_flux_psi_fci",
    "field_alignment_error",
    "sample_psi_along_fieldlines",
    "extract_isosurfaces",
    "fit_flux_surfaces",
    "seed_from_boundary",
    "compute_psi_rz_slices",
    "analyze_flux_surfaces",
    "FluxSurface",
    "PsiSlice",
    "FluxSurfaceAnalysis",
    "plot_psi_along_fieldlines",
    "plot_poincare_with_psi_contours",
    "plot_poincare_overlays",
]
