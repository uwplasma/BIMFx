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

## Planned modules

- `bimfx.vacuum`: stable high-level API (current)
- `bimfx.geometry`: surface construction, normals, orientation, sampling
- `bimfx.io`: import/export (CSV, VMEC `wout`, STL/mesh, checkpoints)
- `bimfx.tracing`: field-line tracing + Poincaré plots
- `bimfx.optim`: thin wrappers to make the solvers ergonomic in optimization loops
- `bimfx.tests`: manufactured solutions and regression tests

## Near-term work items

- Add a checkpoint format (save/load solves)
- Add manufactured-solution tests (sphere / cylinder / torus variants)
- Add exterior vacuum solves consistent with interior solutions
- Provide a small library of example geometries (tokamak / stellarator / mirror / knot)

