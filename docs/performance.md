# Performance Notes

This page summarizes performance characteristics and recommended settings.

## Field-line tracing

- `trace_fieldlines_rk4` (NumPy) is fine for small batches.
- `trace_fieldlines_rk4_jax` can be much faster for large batches and supports GPU/TPU.
  For JAX usage, `B` must be JAX-compatible.

## MFS/BIM solvers

- Cost is dominated by dense boundary interactions.
- Use smaller `k_nn` and moderate `lambda_reg` for quick iterations.
- For production runs, increase boundary sampling and validate convergence.
- JAX solvers are dense and intended for moderate problem sizes.
- BIM supports an optional CG solve with Jacobi preconditioning via `SolveOptions(solver="cg")`.

## Kernel caching

For parameter sweeps where only coefficients change, precompute kernels once:

```python
from bimfx.utils import MFSKernelCache

cache = MFSKernelCache.from_points(X, Y)
phi = cache.phi(alpha)
grad = cache.grad(alpha)
```

## FCI solver

- The sparse CG solve scales with grid size; start with coarse grids (32–64).
- Reduce `eps_perp` to sharpen surfaces, but expect slower convergence.

## Source code

- Tracing backends: [src/bimfx/tracing.py](https://github.com/uwplasma/BIMFx/blob/main/src/bimfx/tracing.py)
