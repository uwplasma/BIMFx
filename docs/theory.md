# Theory (high level)

## Vacuum fields and Laplace’s equation

In a current-free region,

- `∇×B = 0`
- `∇·B = 0`

so we may write `B = ∇φ` with `∇²φ = 0`.

## Boundary condition: flux surface / perfect conductor

Users provide:

- boundary points `x_i` on a closed surface Γ,
- outward unit normals `n_i`.

We enforce:

`n · B = 0` on Γ.

This means magnetic field lines are tangent to Γ.

## Non-uniqueness in toroidal domains

For multiply-connected domains (e.g. a torus), the Neumann problem admits a finite-dimensional family of harmonic-gradient fields (topological degrees of freedom). BIMFx exposes this through a small “multivalued” harmonic basis (toroidal/poloidal) and lets users fix the remaining freedom via flux-like parameters (e.g. `toroidal_flux`).

## Numerical methods

### MFS

Represent the single-valued part of the potential as a sum of fundamental solutions with sources placed on a fictitious surface outside Γ. Enforce the Neumann boundary condition by collocation, with Tikhonov regularization.

### BIM (single-layer)

Represent the single-valued part of the potential as a single-layer potential on Γ. Enforce the Neumann boundary condition using the standard jump relation for the normal derivative and a Nyström discretization over the point cloud, with near-singular regularization.

