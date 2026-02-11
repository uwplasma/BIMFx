import numpy as np

from bimfx.sweep import coarse_to_fine_sweep


def test_coarse_to_fine_sweep_finds_minimum():
    def evaluate(params: dict[str, float]) -> float:
        x = params["x"]
        y = params["y"]
        return (x - 1.5) ** 2 + (y + 0.5) ** 2

    grid = {"x": [0.0, 1.0, 2.0, 3.0], "y": [-2.0, -1.0, 0.0, 1.0]}
    result = coarse_to_fine_sweep(grid, evaluate, levels=2, refine_factor=2)

    best = result.best_params
    assert np.isfinite(result.best_value)
    assert abs(best["x"] - 1.5) < 0.6
    assert abs(best["y"] + 0.5) < 0.6
