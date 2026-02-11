import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from bimfx.pipeline import run_pipeline


def _require_x64() -> None:
    import jax

    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
    if not jax.config.jax_enable_x64:
        pytest.skip("JAX 64-bit mode is required for BIMFx solver tests.")


def test_pipeline_runs(tmp_path: Path) -> None:
    _require_x64()
    rng = np.random.default_rng(0)
    P = rng.normal(size=(60, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    N = P.copy()

    pts_path = tmp_path / "boundary.csv"
    nrm_path = tmp_path / "boundary_normals.csv"
    np.savetxt(pts_path, P, delimiter=",", header="x,y,z", comments="")
    np.savetxt(nrm_path, N, delimiter=",", header="nx,ny,nz", comments="")

    cfg = tmp_path / "pipeline.toml"
    cfg.write_text(
        "\n".join(
            [
                "[boundary]",
                f"input = \"{pts_path}\"",
                f"normals = \"{nrm_path}\"",
                "format = \"csv\"",
                "subsample = 40",
                "",
                "[solve]",
                "method = \"mfs\"",
                "",
                "[validate]",
                "eps_factor = 0.02",
                "",
                "[output]",
                f"dir = \"{tmp_path}\"",
                "",
            ]
        )
    )

    result = run_pipeline(cfg)
    assert "rms" in result.stats
