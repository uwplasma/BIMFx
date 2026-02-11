# Benchmark Gallery

This page summarizes a lightweight timing/accuracy benchmark across bundled datasets.
The numbers are intended for regression tracking and quick comparisons, not for
final production runs. For accuracy studies, see the validation sweep report.

## Summary table

```{include} _static/benchmark/benchmarks.md
```

## Boundary grid (by input type)

![Boundary grid](_static/benchmark/boundary_grid.png)

## Solve time per dataset

![Benchmark time](_static/benchmark/benchmark_time.png)

## Boundary residual per dataset

![Benchmark residual](_static/benchmark/benchmark_rms.png)

## Notes

- All runs use a small subsample of the boundary; increase point counts for production.
- BIM accuracy can depend strongly on `k_nn`, `lambda_reg`, and the near-singular regularization.
- For parameter sweeps, use the coarse-to-fine sweep utilities and the validation report.

## References

- Barnes and Hut (1986) for hierarchical acceleration.
- Greengard and Rokhlin (1987) for the fast multipole method.

See `docs/references.md` for full citations.
