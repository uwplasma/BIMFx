from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


__all__ = ["BarnesHut3D"]


@dataclass
class _Node:
    center: np.ndarray
    half_size: float
    idx: np.ndarray
    mass: float
    centroid: np.ndarray
    children: list["_Node"]

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


def _node_centroid(points: np.ndarray, weights: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if idx.size == 0:
        return np.zeros(3)
    w = weights[idx]
    wsum = float(np.sum(w))
    if abs(wsum) < 1e-12:
        return np.mean(points[idx], axis=0)
    return np.sum(points[idx] * w[:, None], axis=0) / wsum


def _build_node(points: np.ndarray, weights: np.ndarray, idx: np.ndarray, center: np.ndarray, half_size: float, leaf_size: int) -> _Node:
    mass = float(np.sum(weights[idx])) if idx.size > 0 else 0.0
    centroid = _node_centroid(points, weights, idx)
    if idx.size <= leaf_size:
        return _Node(center=center, half_size=half_size, idx=idx, mass=mass, centroid=centroid, children=[])

    children: list[_Node] = []
    hs = half_size * 0.5
    offsets = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    for off in offsets:
        child_center = center + off * hs
        mask = np.all(points[idx] >= (child_center - hs - 1e-15), axis=1) & np.all(
            points[idx] <= (child_center + hs + 1e-15), axis=1
        )
        child_idx = idx[mask]
        if child_idx.size == 0:
            continue
        children.append(_build_node(points, weights, child_idx, child_center, hs, leaf_size))
    return _Node(center=center, half_size=half_size, idx=idx, mass=mass, centroid=centroid, children=children)


class BarnesHut3D:
    """Simple 3D Barnes-Hut tree for 1/r kernels.

    Parameters
    ----------
    points:
        Source points (N,3).
    weights:
        Scalar weights per source (N,).
    theta:
        Opening angle; smaller is more accurate and slower.
    leaf_size:
        Maximum number of sources per leaf.
    """

    def __init__(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        *,
        theta: float = 0.6,
        leaf_size: int = 64,
    ) -> None:
        pts = np.asarray(points, dtype=float)
        wts = np.asarray(weights, dtype=float).reshape(-1)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points shape (N,3), got {pts.shape}")
        if wts.shape[0] != pts.shape[0]:
            raise ValueError("weights must have the same length as points")
        self.points = pts
        self.weights = wts
        self.theta = float(theta)
        self.leaf_size = int(leaf_size)

        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        center = 0.5 * (mins + maxs)
        half_size = 0.5 * float(np.max(maxs - mins) + 1e-12)
        idx = np.arange(pts.shape[0])
        self.root = _build_node(pts, wts, idx, center, half_size, self.leaf_size)

    def _eval_leaf(self, x: np.ndarray, idx: np.ndarray) -> tuple[float, np.ndarray]:
        diff = x[None, :] - self.points[idx]
        r2 = np.sum(diff * diff, axis=1)
        r = np.sqrt(np.maximum(r2, 1e-30))
        G = 1.0 / (4.0 * np.pi * r)
        phi = float(np.sum(G * self.weights[idx]))
        r3 = np.maximum(r2 * r, 1e-30)
        grad = -np.sum(diff * (self.weights[idx] / (4.0 * np.pi * r3))[:, None], axis=0)
        return phi, grad

    def _eval_node(self, x: np.ndarray, node: _Node) -> tuple[float, np.ndarray]:
        if node.is_leaf:
            return self._eval_leaf(x, node.idx)
        dx = x - node.center
        dist = float(np.linalg.norm(dx))
        if dist > 0.0 and node.half_size / dist < self.theta:
            r = max(float(np.linalg.norm(x - node.centroid)), 1e-12)
            phi = node.mass / (4.0 * np.pi * r)
            r3 = max(r**3, 1e-30)
            grad = -(x - node.centroid) * (node.mass / (4.0 * np.pi * r3))
            return float(phi), grad

        phi = 0.0
        grad = np.zeros(3)
        for child in node.children:
            p_c, g_c = self._eval_node(x, child)
            phi += p_c
            grad += g_c
        return phi, grad

    def potential(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return np.array(self._eval_node(X, self.root)[0])
        out = np.empty(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            out[i] = self._eval_node(x, self.root)[0]
        return out

    def gradient(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return np.asarray(self._eval_node(X, self.root)[1])
        out = np.empty_like(X, dtype=float)
        for i, x in enumerate(X):
            out[i] = self._eval_node(x, self.root)[1]
        return out

    def potential_and_gradient(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            phi, grad = self._eval_node(X, self.root)
            return np.array(phi), np.asarray(grad)
        phi = np.empty(X.shape[0], dtype=float)
        grad = np.empty_like(X, dtype=float)
        for i, x in enumerate(X):
            p_i, g_i = self._eval_node(x, self.root)
            phi[i] = p_i
            grad[i] = g_i
        return phi, grad
