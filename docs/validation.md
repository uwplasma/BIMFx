# Validation and Verification

This page summarizes recommended validation checks for BIMFx and provides
helpers you can reuse in your own workflows.

## Boundary condition residuals

For a vacuum field `B = ∇φ`, BIMFx enforces `n·B = 0` on the boundary.
For BIM, evaluate `B` slightly inside the domain (offset along `-n`)
to avoid near-singular evaluation exactly on Γ.

```{math}
r_n = \frac{|n \cdot B|}{\|B\|}
```

```python
import numpy as np
from bimfx.validation import boundary_normal_residual

res = boundary_normal_residual(field.B, P, N, normalize=True)
print("RMS n·B/|B|:", np.sqrt(np.mean(res**2)))
```

## Divergence-free check

```{math}
r_{\nabla\cdot B} = \nabla \cdot B
```

```python
from bimfx.validation import divergence_on_grid, summary_stats

divB = divergence_on_grid(field.B, xs, ys, zs)
print(summary_stats(divB))
```

## End-to-end validation

```python
from bimfx.validation import validate_vacuum_field

stats = validate_vacuum_field(field.B, P, N, xs, ys, zs)
print(stats["boundary_normal_residual"])
print(stats["divergence"])
```

## Recommended reviewer-facing checks

- Report RMS and 95th-percentile `n·B/|B|` on the boundary.
- Report RMS of `∇·B` on an interior grid.
- For toroidal cases, report sensitivity to `k_nn`, `lambda_reg`, and source placement.
- For MFS vs BIM, compare `B` on a fixed probe set.

## Validation sweep report

Generate a paper-ready report (tables + plots) across all bundled inputs:

```bash
JAX_ENABLE_X64=1 python examples/validation_report.py --outdir outputs/validation_report
```

Outputs include:

- `summary.csv` and `summary.md`
- `sweep_k_nn.png`, `sweep_subsample.png`, `sweep_lambda_reg.png`

## Source code

- Validation helpers: [src/bimfx/validation.py](https://github.com/uwplasma/BIMFx/blob/main/src/bimfx/validation.py)
