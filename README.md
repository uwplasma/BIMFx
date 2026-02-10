# BIMFx

Vacuum magnetic field solvers for arbitrary 3D boundaries using:

- **MFS**: Method of Fundamental Solutions
- **BIM**: Boundary integral method (single-layer potential)

Given a closed surface point cloud `x_i` and outward unit normals `n_i`, BIMFx solves Laplace’s equation for a magnetostatic potential `φ` such that the vacuum magnetic field

`B = ∇φ`  

satisfies the **perfect-conductor / flux-surface boundary condition**

`n · B = 0` on the boundary.

The intent is to make *vacuum-field solves differentiable in JAX* so the same code can be used inside geometry optimization loops (e.g. quasi-symmetry design).

## Status

BIMFx is under active development. The current public API focuses on:

- Solving for `B(X)` inside a boundary given `(points, normals)`
- Providing both **MFS** and **BIM** implementations behind a simple interface

Planned functionality (field-line tracing, flux-surface fitting/FCI, manufactured solutions, optimization utilities, VMEC/wout + STL I/O, exterior solves) is tracked in the docs roadmap.

## Install

This is a standard Python package (PEP 517/518). For development:

```bash
git clone https://github.com/uwplasma/BIMFx
cd BIMFx
pip install -e .
```

### JAX 64-bit requirement

BIMFx solvers require JAX **64-bit mode** for numerical stability.

Set the environment variable before running Python:

```bash
export JAX_ENABLE_X64=1
```

## Quickstart

Solve from a boundary point cloud stored in CSV files:

```python
import numpy as np
from bimfx import solve_mfs

P = np.loadtxt("boundary.csv", delimiter=",", skiprows=1)         # (N,3)
N = np.loadtxt("boundary_normals.csv", delimiter=",", skiprows=1) # (N,3)
N = N / np.linalg.norm(N, axis=1, keepdims=True)

field = solve_mfs(P, N, toroidal_flux=1.0)
B = field.B(P)  # (N,3)
```

There is also a small runnable example:

```bash
JAX_ENABLE_X64=1 python examples/solve_vacuum_from_csv.py --help
```

## Boundary I/O

BIMFx includes utilities to load boundary point clouds from common sources:

- VMEC `wout*.nc`
- SLAM `.npz`
- SFLM `.npy`
- STL meshes (optional; requires `trimesh`)

See `docs/io.md` and `examples/convert_boundary.py`.

## Documentation

Documentation lives in `docs/` and is set up for ReadTheDocs.

Build locally:

```bash
pip install -e '.[docs]'
cd docs
make html
```

## Repository layout

- `src/bimfx/`: library code
- `examples/`: runnable scripts
- `inputs/`: small sample boundary datasets
- `docs/`: Sphinx documentation

## License

MIT. See `LICENSE`.
