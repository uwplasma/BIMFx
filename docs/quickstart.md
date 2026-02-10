# Quickstart

## Install (editable)

```bash
git clone https://github.com/uwplasma/BIMFx
cd BIMFx
pip install -e .
```

## Enable JAX 64-bit

BIMFx requires JAX **64-bit mode** for numerical stability:

```bash
export JAX_ENABLE_X64=1
```

## Solve from a boundary point cloud

```python
import numpy as np
from bimfx import solve_mfs

P = np.loadtxt("boundary.csv", delimiter=",", skiprows=1)         # (N,3)
N = np.loadtxt("boundary_normals.csv", delimiter=",", skiprows=1) # (N,3)
N = N / np.linalg.norm(N, axis=1, keepdims=True)

field = solve_mfs(P, N, toroidal_flux=1.0)
B = field.B(P)  # (N,3)
```

## Load boundary from common sources

```python
from bimfx.io import boundary_from_vmec_wout

data = boundary_from_vmec_wout("wout_precise_QA.nc", s=1.0, ntheta=64, nphi=128)
field = solve_mfs(data.points, data.normals, toroidal_flux=1.0)
```

## CLI-style example script

```bash
JAX_ENABLE_X64=1 python examples/solve_vacuum_from_csv.py --method mfs --xyz inputs/knot_tube.csv --normals inputs/knot_tube_normals.csv --subsample 200
```
