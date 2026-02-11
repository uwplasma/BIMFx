"""BIMFx: Vacuum magnetic field solvers in arbitrary 3D geometry.

Primary user-facing API lives in :mod:`bimfx.vacuum`.
"""

from __future__ import annotations

from ._version import __version__
from .vacuum import VacuumField, solve_bim, solve_mfs
from .tracing import (
    FieldlineTrace,
    PoincareSection,
    poincare_sections,
    trace_fieldlines_rk4,
    trace_fieldlines_rk4_jax,
)
from .fci import FCISolution, field_alignment_error, solve_flux_psi_fci
from .jax_solvers import BIMJaxField, MFSJaxField, solve_bim_jax, solve_mfs_jax
from .objectives import boundary_residual_objective, divergence_objective
from .pipeline import run_pipeline, PipelineResult

__all__ = [
    "__version__",
    "VacuumField",
    "solve_bim",
    "solve_mfs",
    "FieldlineTrace",
    "PoincareSection",
    "trace_fieldlines_rk4",
    "trace_fieldlines_rk4_jax",
    "poincare_sections",
    "FCISolution",
    "solve_flux_psi_fci",
    "field_alignment_error",
    "solve_mfs_jax",
    "solve_bim_jax",
    "MFSJaxField",
    "BIMJaxField",
    "boundary_residual_objective",
    "divergence_objective",
    "run_pipeline",
    "PipelineResult",
]
