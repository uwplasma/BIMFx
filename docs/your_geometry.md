# How to Use BIMFx with Your Geometry

This guide shows common pathways for bringing your geometry into BIMFx.

## 1) Mesh (STL/PLY/OBJ)

```python
from bimfx.io import boundary_from_mesh
from bimfx import solve_mfs

data = boundary_from_mesh("your_surface.stl", n_points=6000, even=True, fix_normals=True)
field = solve_mfs(data.points, data.normals, toroidal_flux=1.0)
```

If you prefer the CLI:

```bash
JAX_ENABLE_X64=1 bimfx --input your_surface.stl --method mfs --estimate-normals --validate
```

## 2) Point cloud + estimated normals

```python
import numpy as np
from bimfx.io import estimate_normals
from bimfx import solve_bim

P = np.loadtxt("points.csv", delimiter=",", skiprows=1)
N = estimate_normals(P, k=20)
field = solve_bim(P, N, toroidal_flux=None)
```

## 3) VMEC `wout*.nc`

```python
from bimfx.io import boundary_from_vmec_wout
from bimfx import solve_mfs

data = boundary_from_vmec_wout("wout_precise_QA.nc", s=1.0, ntheta=64, nphi=128)
field = solve_mfs(data.points, data.normals, toroidal_flux=1.0)
```

## 4) Configuration-driven pipeline

Create a simple TOML config:

```toml
[boundary]
input = "inputs/wout_precise_QA.nc"
format = "nc"
subsample = 800

[solve]
method = "bim"
toroidal_flux = 1.0
acceleration = "barnes-hut"
accel_theta = 0.6

[validate]
eps_factor = 0.02

[output]
dir = "outputs/pipeline"
```

Run it:

```bash
python -c "from bimfx.pipeline import run_pipeline; run_pipeline('your_config.toml')"
```

## Tips

- Use `k_nn` and `lambda_reg` sweeps to tune accuracy and stability.
- Enable Barnes–Hut acceleration for large point sets.
- Always check `n·B` residuals and compare BIM vs MFS for sanity.
