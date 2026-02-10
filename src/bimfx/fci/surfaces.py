from __future__ import annotations

from typing import Iterable

import numpy as np


def extract_isosurfaces(
    psi: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    levels: Iterable[float],
) -> list[dict[str, np.ndarray]]:
    """Extract isosurfaces using marching cubes.

    Returns a list of dicts with keys: vertices, faces, level.
    Requires `scikit-image`.
    """
    try:
        from skimage.measure import marching_cubes
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-image is required for isosurface extraction.") from exc

    surfaces = []
    for level in levels:
        verts, faces, _normals, _values = marching_cubes(psi, level=level)
        # Map from index space to physical space
        vx = np.interp(verts[:, 0], np.arange(len(xs)), xs)
        vy = np.interp(verts[:, 1], np.arange(len(ys)), ys)
        vz = np.interp(verts[:, 2], np.arange(len(zs)), zs)
        verts_phys = np.stack([vx, vy, vz], axis=1)
        surfaces.append({"level": float(level), "vertices": verts_phys, "faces": faces})
    return surfaces

