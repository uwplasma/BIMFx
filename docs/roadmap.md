# Roadmap

This repository is being consolidated from an earlier “LX” umbrella repository into a focused, self-contained vacuum-field solver package.

## Intended user experience

1. User provides a boundary as either:
   - a point cloud + normals, or
   - a structured surface object (VMEC, Fourier surface, STL, …).
2. User calls a solver (MFS or BIM) with one line of code.
3. The solver returns a callable `B(X)` (and optional `φ(X)`), compatible with:
   - field-line tracing,
   - objective evaluation for optimization,
   - automatic differentiation in JAX.

## I/O conventions (planned)

- The **library** should not assume any particular working directory.
- Example scripts may use a conventional layout:
  - `inputs/` for boundary files and VMEC outputs
  - `outputs/` for checkpoints, figures, and logs (ignored by git)
- Checkpoints should be *portable* and allow:
  - reloading a solution to evaluate `B(X)` without re-solving
  - resuming solves (when supported)
  - pairing interior/exterior solves consistently

## Planned modules

- `bimfx.vacuum`: stable high-level API (current)
- `bimfx.geometry`: surface construction, normals, orientation, sampling
- `bimfx.io`: import/export (CSV, VMEC `wout`, STL/mesh, checkpoints)
- `bimfx.tracing`: field-line tracing + Poincaré plots
- `bimfx.optim`: thin wrappers to make the solvers ergonomic in optimization loops
- `bimfx.tests`: manufactured solutions and regression tests

## Functionality inventory (targets)

- **Vacuum solves**
  - interior MFS (current)
  - interior BIM (current)
  - exterior MFS/BIM consistent with an interior solution (planned)
  - robust handling of multiply-connected topology (flux constraints / basis selection) (planned)
- **Diagnostics**
  - boundary-condition residuals evaluated on an interior offset surface (current)
  - Laplacian residual checks with manufactured solutions (planned)
  - energy / flux functionals suitable for optimization (planned)
- **Field-line tracing**
  - fast RK integrators + adaptive stepping (planned)
  - Poincaré sections and connection-length diagnostics (planned)
- **Geometry + I/O**
  - CSV boundary point clouds (current)
  - VMEC `wout` → boundary extraction (current)
  - STL/mesh → point cloud + normals (current, optional `trimesh`)
  - built-in example geometries (tokamak/stellarator/mirror/knot) (planned)
- **Optimization**
  - objective wrappers (boundary/divergence objectives) (current)
  - compatibility with `jaxopt`/`optax` optimization loops (planned)

## Near-term work items

- Add a checkpoint format (save/load solves)
- Add manufactured-solution tests (sphere / cylinder / torus variants)
- Add exterior vacuum solves consistent with interior solutions
- Provide a small library of example geometries (tokamak / stellarator / mirror / knot)
- Explore fast multipole / kernel-acceleration for large boundary sets
