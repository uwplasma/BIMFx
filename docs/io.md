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

## VMEC `wout*.nc`

```python
from bimfx.io import boundary_from_vmec_wout

data = boundary_from_vmec_wout("wout_precise_QA.nc", s=1.0, ntheta=64, nphi=128)
```

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

