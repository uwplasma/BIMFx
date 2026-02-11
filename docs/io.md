# Boundary I/O

This module provides utilities to load and export boundary point clouds and normals
in the formats used throughout the legacy LX repository.

## CSV

```python
from bimfx.io import load_boundary_csv, save_boundary_csv

data = load_boundary_csv("boundary.csv", "boundary_normals.csv")
points, normals = data.points, data.normals

save_boundary_csv(points, normals, "boundary_export")
```

## Sample input datasets

The repository ships small reference datasets under `inputs/`:

- VMEC boundary (NetCDF): `inputs/wout_precise_QA.nc`
- VMEC boundary (NetCDF): `inputs/wout_SLAM_6_coils.nc`
- Near-axis boundary (CSV + normals): `inputs/wout_LandremanSenguptaPlunk_5.3.csv`
- Mirror boundary (CSV + normals): `inputs/sflm_rm4.csv`
- Knot boundary (CSV + normals): `inputs/knot_tube.csv`

Example usage from CSV:

```bash
JAX_ENABLE_X64=1 python examples/solve_vacuum_from_csv.py --method mfs --xyz inputs/knot_tube.csv --normals inputs/knot_tube_normals.csv --subsample 200
```

Example usage from VMEC `wout*.nc`:

```bash
JAX_ENABLE_X64=1 python examples/solve_from_vmec_wout.py --wout inputs/wout_precise_QA.nc --method bim
```

## VMEC `wout*.nc`

```python
from bimfx.io import boundary_from_vmec_wout

data = boundary_from_vmec_wout("wout_precise_QA.nc", s=1.0, ntheta=64, nphi=128)
```

VMEC reference: Hirshman & Whitson (1983),
[DOI: 10.1063/1.864116](https://doi.org/10.1063/1.864116).

## SLAM `.npz`

```python
from bimfx.io import boundary_from_slam_npz

data = boundary_from_slam_npz("slam_surface.npz", ntheta=64, nphi=128)
```

## SFLM `.npy`

```python
from bimfx.io import boundary_from_sflm_npy

data = boundary_from_sflm_npy("sflm_surface.npy")
```

## STL meshes

```python
from bimfx.io import boundary_from_stl

data = boundary_from_stl("coil_shell.stl", n_points=5000, even=True)
```

Requires `trimesh`:

```bash
pip install "bimfx[io]"
```

## Source code

- Boundary loaders: [src/bimfx/io/boundary.py](https://github.com/uwplasma/BIMFx/blob/main/src/bimfx/io/boundary.py)
