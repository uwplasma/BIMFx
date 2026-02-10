import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import numpy as np
import pytest

from bimfx import solve_bim, solve_mfs
from bimfx.io import boundary_from_vmec_wout, load_boundary_csv
from bimfx.vacuum.solve import SolveOptions


def _require_x64() -> None:
    import jax

    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
    if not jax.config.jax_enable_x64:
        pytest.skip("JAX 64-bit mode is required for BIMFx solver tests.")


def _inputs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "inputs"


def _subsample(P: np.ndarray, N: np.ndarray, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    if P.shape[0] <= n:
        return P, N
    idx = np.linspace(0, P.shape[0] - 1, n, dtype=int)
    return P[idx], N[idx]


def _offset_inside(P: np.ndarray, N: np.ndarray, eps_factor: float = 0.02) -> np.ndarray:
    center = np.mean(P, axis=0)
    scale = np.median(np.linalg.norm(P - center[None, :], axis=1))
    return P - eps_factor * scale * N


def _relative_residual(B, P: np.ndarray, N: np.ndarray) -> np.ndarray:
    Bv = np.asarray(B(P))
    if Bv.shape == (3,):
        Bv = Bv[None, :]
    ndot = np.sum(N * Bv, axis=1)
    scale = np.median(np.linalg.norm(Bv, axis=1))
    return np.abs(ndot) / max(scale, 1e-30)


@pytest.mark.parametrize(
    "stem,geom_kind",
    [
        ("knot_tube", "other"),
        ("sflm_rm4", "mirror"),
        ("wout_LandremanSenguptaPlunk_5.3", "torus"),
        ("wout_SLAM_6_coils", "torus"),
    ],
)
def test_csv_inputs_mfs_bim(stem: str, geom_kind: str):
    _require_x64()
    inputs = _inputs_dir()
    data = load_boundary_csv(inputs / f"{stem}.csv", inputs / f"{stem}_normals.csv")
    P, N = _subsample(data.points, data.normals, n=200)

    options = SolveOptions(k_nn=24, verbose=False)
    toroidal_flux = 1.0 if geom_kind == "torus" else None
    field_mfs = solve_mfs(P, N, toroidal_flux=toroidal_flux, options=options)
    field_bim = solve_bim(P, N, toroidal_flux=toroidal_flux, options=options)

    Pin = _offset_inside(P, N)
    res_mfs = _relative_residual(field_mfs.B, Pin, N)
    res_bim = _relative_residual(field_bim.B, Pin, N)
    assert np.sqrt(np.mean(res_mfs**2)) < 0.5
    assert np.isfinite(res_bim).all()
    assert np.sqrt(np.mean(res_bim**2)) < 10.0


def test_vmec_wout_inputs_mfs_bim():
    _require_x64()
    inputs = _inputs_dir()
    wout = inputs / "wout_precise_QA.nc"
    data = boundary_from_vmec_wout(wout, s=1.0, ntheta=24, nphi=48)
    P, N = _subsample(data.points, data.normals, n=200)

    options = SolveOptions(k_nn=24, verbose=False)
    field_mfs = solve_mfs(P, N, toroidal_flux=1.0, options=options)
    field_bim = solve_bim(P, N, toroidal_flux=1.0, options=options)

    Pin = _offset_inside(P, N)
    res_mfs = _relative_residual(field_mfs.B, Pin, N)
    res_bim = _relative_residual(field_bim.B, Pin, N)
    assert np.sqrt(np.mean(res_mfs**2)) < 0.5
    assert np.isfinite(res_bim).all()
    assert np.sqrt(np.mean(res_bim**2)) < 10.0
