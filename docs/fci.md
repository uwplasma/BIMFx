# Flux-Surface Finding (FCI)

This module implements a **flux-coordinate independent (FCI)-inspired** solver for a
flux-like scalar `psi` by solving a strongly anisotropic diffusion problem aligned
with the vacuum field `B = ∇φ`.

## Solve for `psi`

```python
import numpy as np
from bimfx.fci import solve_flux_psi_fci

def B(points):
    # Example: uniform field along z
    return np.tile([0.0, 0.0, 1.0], (points.shape[0], 1))

# Boundary point cloud + normals
P = np.loadtxt("boundary.csv", delimiter=",", skiprows=1)
N = np.loadtxt("boundary_normals.csv", delimiter=",", skiprows=1)

sol = solve_flux_psi_fci(B, P, N, nx=48, ny=48, nz=48)
psi = sol.psi
```

## Diagnostics

```python
from bimfx.fci import field_alignment_error

q = field_alignment_error(sol.psi, sol.xs, sol.ys, sol.zs, B)
```

## Isosurface extraction

```python
from bimfx.fci import extract_isosurfaces

surfaces = extract_isosurfaces(sol.psi, sol.xs, sol.ys, sol.zs, levels=[0.2, 0.4, 0.6])
```

Requires:

```bash
pip install "bimfx[fci]"
```

