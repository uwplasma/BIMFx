"""BIMFx: Vacuum magnetic field solvers in arbitrary 3D geometry.

Primary user-facing API lives in :mod:`bimfx.vacuum`.
"""

from __future__ import annotations

from ._version import __version__
from .vacuum import VacuumField, solve_bim, solve_mfs

__all__ = [
    "__version__",
    "VacuumField",
    "solve_bim",
    "solve_mfs",
]
