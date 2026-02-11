# BIMFx

**BIMFx** provides vacuum magnetic field solvers for arbitrary 3D boundaries using:

- the **Method of Fundamental Solutions (MFS)**, and
- a **Boundary Integral Method (BIM)** based on a single-layer potential.

The physical target is a vacuum magnetic field

`B = ∇φ`, with `∇²φ = 0`,

inside a closed surface, enforcing the flux-surface / perfect-conductor boundary condition

`n · B = 0` on the boundary.

```{toctree}
:maxdepth: 2

quickstart
theory
io
tracing
fci
validation
performance
examples
references
differentiable
workflow
api
roadmap
```
