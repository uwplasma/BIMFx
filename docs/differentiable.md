# Differentiable Core (JAX)

BIMFx provides JAX-native solvers that are fully differentiable with respect to
boundary points and normals. This enables gradient-based optimization of
geometry parameters (e.g. coil shapes or boundary perturbations).

## JAX MFS and BIM

```python
import jax
import jax.numpy as jnp
from bimfx import solve_mfs_jax, solve_bim_jax
from bimfx.objectives import boundary_residual_objective

# P, N: boundary points + normals
field = solve_mfs_jax(P, N, harmonic_coeffs=(1.0, 0.0))
loss = boundary_residual_objective(field.B, P, N)

grad_P = jax.grad(lambda Pvar: boundary_residual_objective(
    solve_bim_jax(Pvar, N, harmonic_coeffs=(1.0, 0.0)).B, Pvar, N))(P)
```

## Notes

- The JAX solvers are dense and intended for moderate problem sizes.
- Multivalued toroidal components are supported via `harmonic_coeffs=(a_t, 0.0)`.
- BIM JAX gradients can be sensitive to regularization; increase `lambda_reg` for stability.
- Linear solves use implicit differentiation (custom VJP) for stable gradients.
- Use `stop_gradient=True` only if you want to disable gradients through the solve.
- `stop_gradient_knn=True` (default) avoids non-smooth gradients from kNN radius selection.

## Further reading

- See `docs/references.md` for citations on implicit differentiation, MFS/BIM, and fast multipole methods.
- For toroidal domains, pass `harmonic_coeffs=(a_t, 0.0)` to specify toroidal flux.
