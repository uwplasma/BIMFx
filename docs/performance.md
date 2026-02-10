# Performance Notes

This page summarizes performance characteristics and recommended settings.

## Field-line tracing

- `trace_fieldlines_rk4` (NumPy) is fine for small batches.
- `trace_fieldlines_rk4_jax` can be much faster for large batches and supports GPU/TPU.

## MFS/BIM solvers

- Cost is dominated by dense boundary interactions.
- Use smaller `k_nn` and moderate `lambda_reg` for quick iterations.
- For production runs, increase boundary sampling and validate convergence.

## FCI solver

- The sparse CG solve scales with grid size; start with coarse grids (32–64).
- Reduce `eps_perp` to sharpen surfaces, but expect slower convergence.
