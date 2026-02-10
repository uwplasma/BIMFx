# Field-Line Tracing and Poincare Sections

The tracing tools operate on a user-supplied magnetic field callable `B(X)`.

## Trace field lines

```python
import numpy as np
from bimfx import trace_fieldlines_rk4

def B(points):
    # Example: rotation around z (circles in xy)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.stack([-y, x, 0.0 * z], axis=1)

seeds = np.array([[1.0, 0.0, 0.0]])
trace = trace_fieldlines_rk4(B, seeds, ds=0.05, n_steps=400, normalize=True)
traj = trace.trajectories
```

### JAX backend

For large batches, the JAX backend can be faster (and GPU/TPU compatible):

```python
from bimfx import trace_fieldlines_rk4_jax

trace = trace_fieldlines_rk4_jax(B, seeds, ds=0.05, n_steps=400, normalize=True)
```

Note: `B` must be JAX-compatible for this backend.

## Poincare sections

```python
from bimfx import poincare_sections

sec = poincare_sections(traj, phi_planes=[0.0], nfp=1)
R, Z = sec.R, sec.Z
```

`phi_planes` are in cylindrical coordinates and repeated every `2π/nfp`.
