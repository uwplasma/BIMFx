from __future__ import annotations

from .geometry import (
    ScaleInfo,
    best_fit_axis,
    detect_geometry_and_axis,
    kNN_geometry_stats,
    maybe_flip_normals,
    multivalued_bases_about_axis,
    normalize_geometry,
    orthonormal_complement,
    project_to_local,
)
from .solver import solve_mfs_neumann
from .sources_kernels import build_evaluators_mfs

__all__ = [
    "ScaleInfo",
    "best_fit_axis",
    "detect_geometry_and_axis",
    "kNN_geometry_stats",
    "maybe_flip_normals",
    "multivalued_bases_about_axis",
    "normalize_geometry",
    "orthonormal_complement",
    "project_to_local",
    "solve_mfs_neumann",
    "build_evaluators_mfs",
]
