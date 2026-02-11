import numpy as np

from bimfx.utils.fastsum import BarnesHut3D


def _direct_potential(points: np.ndarray, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - points[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    r = np.maximum(r, 1e-12)
    G = 1.0 / (4.0 * np.pi * r)
    return G @ weights


def _direct_gradient(points: np.ndarray, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - points[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    r3 = np.maximum(r2 * np.sqrt(r2), 1e-30)
    grad = -diff / (4.0 * np.pi * r3[..., None])
    return np.sum(grad * weights[None, :, None], axis=1)


def test_barnes_hut_accuracy():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(200, 3))
    weights = rng.normal(size=(200,))
    X = rng.normal(size=(40, 3))

    tree = BarnesHut3D(points, weights, theta=0.6, leaf_size=32)
    phi_bh = tree.potential(X)
    grad_bh = tree.gradient(X)

    phi_ref = _direct_potential(points, weights, X)
    grad_ref = _direct_gradient(points, weights, X)

    phi_err = np.linalg.norm(phi_bh - phi_ref) / np.maximum(1e-12, np.linalg.norm(phi_ref))
    grad_err = np.linalg.norm(grad_bh - grad_ref) / np.maximum(1e-12, np.linalg.norm(grad_ref))

    assert phi_err < 0.15
    assert grad_err < 0.2
