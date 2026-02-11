from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SweepRecord:
    params: dict[str, float]
    value: float
    level: int


@dataclass(frozen=True)
class SweepResult:
    records: list[SweepRecord]
    best_params: dict[str, float]
    best_value: float


def coarse_to_fine_sweep(
    grid: dict[str, list[float]],
    evaluate: Callable[[dict[str, float]], float],
    *,
    levels: int = 2,
    refine_factor: int = 2,
) -> SweepResult:
    """Run a coarse-to-fine parameter sweep and return the best point."""
    if not grid:
        raise ValueError("grid must have at least one parameter")

    current = {k: sorted(float(v) for v in vals) for k, vals in grid.items()}
    records: list[SweepRecord] = []
    best_params: dict[str, float] = {}
    best_value = float("inf")

    for level in range(levels):
        combos = list(product(*[current[k] for k in current.keys()]))
        for combo in combos:
            params = {k: float(v) for k, v in zip(current.keys(), combo)}
            val = float(evaluate(params))
            records.append(SweepRecord(params=params, value=val, level=level))
            if val < best_value:
                best_value = val
                best_params = params

        # refine around the best parameters
        refined: dict[str, list[float]] = {}
        for key, values in current.items():
            vals = np.array(values, dtype=float)
            best = best_params.get(key, vals[len(vals) // 2])
            idx = int(np.argmin(np.abs(vals - best)))
            lo = vals[max(idx - 1, 0)]
            hi = vals[min(idx + 1, len(vals) - 1)]
            if lo == hi:
                refined[key] = list(vals)
                continue
            n_new = max(3, refine_factor * 2 + 1)
            refined_vals = np.linspace(lo, hi, n_new)
            refined[key] = sorted(set(np.concatenate([vals, refined_vals]).tolist()))
        current = refined

    return SweepResult(records=records, best_params=best_params, best_value=best_value)


__all__ = ["SweepRecord", "SweepResult", "coarse_to_fine_sweep"]
