import numpy as np

from bimfx.utils import MFSKernelCache


def test_kernel_cache_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    Y = rng.normal(size=(12, 3))
    cache = MFSKernelCache.from_points(X, Y)
    assert cache.G.shape == (10, 12)
    assert cache.gradG.shape == (10, 12, 3)
