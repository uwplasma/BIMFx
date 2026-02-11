# Geometry Preprocessing Guide

This guide outlines best practices for preparing meshes and point clouds for BIMFx.
The goal is to produce consistent, outward-facing normals and well-sampled surfaces.

## Mesh preprocessing checklist

1. Ensure the mesh is watertight (no holes or self-intersections).
2. Normalize units and coordinate conventions (meters recommended).
3. Remove duplicate vertices and fix flipped faces.
4. Sample points uniformly (or curvature-aware if needed).

BIMFx provides mesh sampling helpers:

```python
from bimfx.io import boundary_from_mesh

data = boundary_from_mesh("your_surface.stl", n_points=4096, even=True, fix_normals=True)
P, N = data.points, data.normals
```

## Point cloud preprocessing

If you only have points, BIMFx can estimate normals via local PCA:

```python
from bimfx.io import estimate_normals

N = estimate_normals(P, k=20)
```

For noisy point clouds, increase `k` and smooth the point cloud beforehand.
Normal estimation references include Hoppe et al. (1992) and variants of local PCA.

## Consistent normal orientation rules

Many workflows fail because normals are inconsistently oriented. Use these checks:

- **Outward test**: compute the centroid `c = mean(P)`, then ensure
  `dot(P - c, N) > 0` for most points.
- **Flip if needed**: if the average dot product is negative, flip all normals.
- **Spot check**: visualize a subset of normals to verify orientation.

BIMFx applies a centroid-based outward check automatically and records whether
normals were flipped.

## Schema versioning and provenance

`BoundaryData` carries schema and provenance metadata to make datasets reproducible:

- `schema_version`: current version is `1.0`
- `provenance`: source path, sampling parameters, and loader metadata

Use this to document how a boundary was created before optimization.

## References

- Hoppe et al., "Surface Reconstruction from Unorganized Points" (1992). [PDF](https://hhoppe.com/recon.pdf)
- Curless and Levoy, "A Volumetric Method for Building Complex Models from Range Images" (1996). [PDF](https://graphics.stanford.edu/papers/volrange/volrange.pdf)
- Kazhdan et al., "Poisson Surface Reconstruction" (2006). [PDF](https://www.cs.jhu.edu/~misha/MyPapers/SGP06.pdf)
