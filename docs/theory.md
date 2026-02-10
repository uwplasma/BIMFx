# Theory (high level)

This section collects the core equations and discretizations used in BIMFx.

## Vacuum fields and Laplace’s equation

In a current-free region,

- `∇×B = 0`
- `∇·B = 0`

so we may write `B = ∇φ` with

$$
\nabla^2 \phi = 0.
$$

## Boundary condition: flux surface / perfect conductor

Users provide:

- boundary points `x_i` on a closed surface Γ,
- outward unit normals `n_i`.

We enforce

$$
n \cdot B = n \cdot \nabla \phi = 0 \quad \text{on } \Gamma,
$$

so that magnetic field lines are tangent to Γ.

## Non-uniqueness in toroidal domains

For multiply-connected domains (e.g. a torus), the Neumann problem admits a finite-dimensional family of harmonic-gradient fields. BIMFx exposes this through a small multivalued harmonic basis (toroidal/poloidal) and lets users fix the remaining freedom via flux-like parameters (e.g. `toroidal_flux`).

## Numerical methods

### Method of Fundamental Solutions (MFS)

The single-valued part of the potential is represented as

$$
\phi_s(x) = \sum_{j=1}^M \alpha_j G(x, y_j),
$$

with the free-space Green’s function

$$
G(x, y) = \frac{1}{4\pi \|x - y\|}.
$$

Sources `y_j` are placed on a fictitious surface outside Γ, and the Neumann condition is enforced by collocation with Tikhonov regularization. See, e.g., the classical MFS reference by Karageorghis & Fairweather (1989) [DOI: 10.1093/imanum/9.2.231](https://doi.org/10.1093/imanum/9.2.231).

### Boundary Integral Method (single-layer)

The single-layer potential is

$$
\phi_s(x) = \int_{\Gamma} G(x, y)\,\sigma(y)\,dS_y,
$$

and the interior Neumann trace satisfies the jump relation

$$
\partial_n \phi_s|_{\Gamma^-} = \left(-\tfrac{1}{2}I + K'\right)\sigma,
$$

where `K'` is the adjoint double-layer operator. BIMFx discretizes this with a Nyström method over the boundary point cloud, and adds a near-singular regularization. For background, see Kress (2014) [DOI: 10.1007/978-1-4614-9593-2](https://doi.org/10.1007/978-1-4614-9593-2).

## FCI-inspired flux-surface solve

To recover a flux-like scalar `ψ` aligned with `B`, BIMFx solves the strongly anisotropic diffusion equation

$$
\nabla \cdot (D \nabla \psi) = 0, \qquad
D = \epsilon_\perp I + (1-\epsilon_\perp)\, t t^\top, \quad t = \frac{B}{\|B\|}.
$$

This is inspired by flux-coordinate independent (FCI) approaches used in plasma simulation (e.g. Harvey et al. 2013) [DOI: 10.1016/j.cpc.2013.06.005](https://doi.org/10.1016/j.cpc.2013.06.005).

## Source code pointers

- MFS solver: [src/bimfx/mfs/solver.py](https://github.com/uwplasma/BIMFx/blob/main/src/bimfx/mfs/solver.py)
- BIM solver: [src/bimfx/bim/solver.py](https://github.com/uwplasma/BIMFx/blob/main/src/bimfx/bim/solver.py)
- Vacuum solver API: [src/bimfx/vacuum/solve.py](https://github.com/uwplasma/BIMFx/blob/main/src/bimfx/vacuum/solve.py)
- FCI solver: [src/bimfx/fci/solve.py](https://github.com/uwplasma/BIMFx/blob/main/src/bimfx/fci/solve.py)
