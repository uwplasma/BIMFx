from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from bimfx.bim.solver import solve_bim_neumann
from bimfx.mfs.solver import solve_mfs_neumann
from bimfx.vacuum.field import VacuumField


@dataclass(frozen=True)
class SolveOptions:
    method: Literal["mfs", "bim"] = "mfs"
    use_multivalued: bool = True
    k_nn: int = 48
    source_factor: float = 2.0
    lambda_reg: float = 1e-6
    clip_factor: float = 0.2  # BIM near-singular regularization
    solver: Literal["direct", "cg"] = "direct"
    cg_tol: float = 1e-8
    cg_maxiter: int = 500
    preconditioner: Literal["jacobi", "none"] = "jacobi"
    verbose: bool = True


def solve_mfs(
    points: Any,
    normals: Any,
    *,
    harmonic_coeffs: tuple[float, float] | None = None,
    toroidal_flux: float | None = None,
    poloidal_flux: float | None = None,
    options: SolveOptions | None = None,
) -> VacuumField:
    """Solve for a vacuum field using the Method of Fundamental Solutions (MFS).

    The solver enforces the boundary condition ``n·B = 0`` on the provided
    point cloud (with outward unit normals), returning a callable magnetic
    field ``B(X)`` inside the domain.

    Notes
    -----
    In a toroidal domain, the Neumann problem admits a 2D family of harmonic
    gradient fields. You can fix that freedom by specifying either:

    - ``harmonic_coeffs=(a_t, a_p)`` directly, or
    - ``toroidal_flux=Φ_t`` (sets ``a_t = Φ_t/(2π)`` and ``a_p=0``).
    """
    if options is None:
        options = SolveOptions(method="mfs")
    if poloidal_flux is not None:
        raise NotImplementedError("poloidal_flux is not yet wired; pass harmonic_coeffs=(a_t,a_p).")

    if harmonic_coeffs is None and toroidal_flux is not None:
        harmonic_coeffs = (float(toroidal_flux) / (2.0 * np.pi), 0.0)

    solution = solve_mfs_neumann(
        points,
        normals,
        use_multivalued=options.use_multivalued,
        k_nn=options.k_nn,
        source_factor=options.source_factor,
        lambda_reg=options.lambda_reg,
        harmonic_coeffs=harmonic_coeffs,
        verbose=options.verbose,
    )

    return VacuumField(
        phi=solution.phi,
        B=solution.B,
        metadata=solution.metadata,
    )


def solve_bim(
    points: Any,
    normals: Any,
    *,
    harmonic_coeffs: tuple[float, float] | None = None,
    toroidal_flux: float | None = None,
    poloidal_flux: float | None = None,
    options: SolveOptions | None = None,
) -> VacuumField:
    """Solve for a vacuum field using a boundary integral method (BIM)."""
    if options is None:
        options = SolveOptions(method="bim")
    if poloidal_flux is not None:
        raise NotImplementedError("poloidal_flux is not yet wired; pass harmonic_coeffs=(a_t,a_p).")

    if harmonic_coeffs is None and toroidal_flux is not None:
        harmonic_coeffs = (float(toroidal_flux) / (2.0 * np.pi), 0.0)

    solution = solve_bim_neumann(
        points,
        normals,
        use_multivalued=options.use_multivalued,
        k_nn=options.k_nn,
        lambda_reg=options.lambda_reg,
        clip_factor=options.clip_factor,
        harmonic_coeffs=harmonic_coeffs,
        solver=options.solver,
        cg_tol=options.cg_tol,
        cg_maxiter=options.cg_maxiter,
        preconditioner=options.preconditioner,
        verbose=options.verbose,
    )
    return VacuumField(phi=solution.phi, B=solution.B, metadata=solution.metadata)
