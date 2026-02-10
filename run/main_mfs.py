
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## Run as python main.py --sf_min 0.5 --lbfgs-maxiter 50 --k-nn 128 for high-accuracy solve
## Run as python main.py --sf_min 1.5 --lbfgs-maxiter 5 --k-nn 32 for fast solve
### Example python main.py wout_precise_QA.csv wout_precise_QA_normals.csv
### SLAM surface might need --sf_min 3.0 for stability
"""
Laplace (∇²φ = 0) inside a closed 3D surface with Neumann BC n·∇φ = 0 (total)
via Method of Fundamental Solutions (MFS) with multivalued pieces (toroidal+poloidal).

Representation:
  φ(x) = φ_mv(x) + ψ(x),  ψ(x) = Σ_j α_j G(x, y_j),   G = 1/(4π|x - y|)
where the sources y_j are on an OUTER fictitious surface built by offsetting the
boundary points along outward normals by a distance δ.

We solve a Tikhonov-regularized weighted least-squares for [α, a_t, a_p]:
  minimize || W^{1/2} ( A α + D a - g ) ||_2^2 + λ^2 ( ||α||_2^2 + γ^2 ||a||_2^2 )
where:
  - A_ij = n_i · ∇_x G(x_i, y_j)
  - D has 2 columns: D[:,0] = n·grad_t(x_i), D[:,1] = n·grad_p(x_i)
  - g = 0  (we solve for total Neumann = 0)   equivalently Aα + D a ≈ 0
    (this automatically cancels the multivalued normal component)
We enforce compatibility by centering D columns under W (optional but helpful).

Diagnostics and plots:
  • Rich prints: kNN scales, δ choice, λ choice, system sizes, condition surrogates,
    residual norms, |∇φ| stats on Γ and on Γ₋ (interior-offset ring), flux neutrality on Γ₋,
    and |∇²ψ| near boundary (should be tiny).
  • Plots: geometry+normals, |∇φ| on Γ₋, BC error |n·∇φ| on Γ₋, Laplacian histogram.

Inputs:
  slam_surface.csv           columns: x,y,z
  slam_surface_normals.csv   columns: nx,ny,nz

Deps: jax (64-bit), jaxlib, numpy, matplotlib, scikit-learn
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import argparse
import os
from pathlib import Path
script_dir = Path(__file__).resolve().parent

from bimfx.utils.io import get_candidates, load_surface_xyz_normals
from bimfx.utils.plotting import (plot_geometry_and_solution, plot_boundary_condition_errors,
                                 plot_laplacian_errors_on_interior_band)
from bimfx.utils.printing import vec_stats
from bimfx.mfs.solvers import (autotune, build_ring_weights, build_system_matrices,
                               fit_mv_coeffs_minimize_rhs, make_flux_constraint,
                               build_A_rows_at_points, build_cap_flux_constraint,
                               augmented_lagrangian_solve)
from bimfx.mfs.geometry import (normalize_geometry, kNN_geometry_stats, maybe_flip_normals)
from bimfx.mfs.sources_kernels import build_evaluators_mfs

# file_name = "wout_precise_QA"
# file_name = "wout_precise_QH"
# file_name = "wout_SLAM_6_coils"
# file_name = "wout_SLAM_4_coils"
# file_name = "sflm_rm4"
file_name = "knot_tube"

# ------------------------------- main ------------------------------- #
def main(xyz_csv="slam_surface.csv", nrm_csv="slam_surface_normals.csv",
         use_mv=True, k_nn=48,
         mv_weight=0.5,               # regularization weight for [a_t,a_p]
         interior_eps_factor=5e-3,  # ε ~ interior offset for evaluation, in *normalized* h units
         verbose=True, show_normals=False,
         toroidal_flux=None,          # prescribe Φ_t (sets a_t = Φ_t/(2π)) if not None
         sf_min=1.0, sf_max=6.5, lbfgs_maxiter=30, lbfgs_tol=1e-8,
         geom_kind="auto", bc_weight=20.0
        ):

    # Load & orient
    P, N = load_surface_xyz_normals(xyz_csv, nrm_csv, verbose=verbose)
    print(f"[INFO] Npoints={P.shape[0]}")
    N, flipped = maybe_flip_normals(P, N)

    # Normalize geometry (for numerics)
    Pn, scinfo = normalize_geometry(P, verbose=verbose)
    Nn = N  # direction unchanged by uniform scaling/translation for normals

    # kNN scales & crude surface weights
    W, rk = kNN_geometry_stats(Pn, k=k_nn, verbose=verbose)
    h_med = float(np.median(rk))
    print(f"[SCALE] median local spacing h_med≈{h_med:.4g} (normalized units)")
    
    # --- Auto-tune ---
    best = autotune(P, N, Pn, Nn, W, rk, scinfo, use_mv=use_mv, mv_weight=mv_weight,
                    interior_eps_factor=interior_eps_factor, verbose=True,
                    sf_min=sf_min, sf_max=sf_max, bc_weight=bc_weight,
                    lbfgs_maxiter=lbfgs_maxiter, lbfgs_tol=lbfgs_tol,
                    geom_kind=geom_kind)
    source_factor_opt = best["source_factor"]
    lambda_reg_opt    = best["lambda_reg"]
    # Reuse pre-built things to avoid recompute:
    phi_t, grad_t, phi_p, grad_p, Yn, kind = best["phi_t"], best["grad_t"], best["phi_p"], best["grad_p"], best["Yn"], best["kind"]
    delta_n = float(np.median(np.linalg.norm(np.asarray(Yn) - np.asarray(Pn), axis=1)))

    A, D = build_system_matrices(Pn, Nn, Yn, W, grad_t, grad_p, scinfo,
                                 use_mv=use_mv, center_D=True, verbose=verbose)

    if use_mv:
        grad_t_bdry, grad_p_bdry = grad_t(Pn), grad_p(Pn)
        a, D_raw, D0 = fit_mv_coeffs_minimize_rhs(Nn, W, grad_t_bdry, grad_p_bdry, verbose=verbose)
        g_raw = scinfo.scale * jnp.sum(Nn * (a[0]*grad_t_bdry + a[1]*grad_p_bdry), axis=1)
    else:
        a = jnp.zeros((2,), dtype=jnp.float64)
        g_raw = jnp.zeros((Pn.shape[0],), dtype=jnp.float64)

    # --- OVERRIDE a_t if a toroidal flux is prescribed ---
    if use_mv and (toroidal_flux is not None):
        a_t_fixed = float(toroidal_flux) / (2.0 * np.pi)
        a = jnp.array([a_t_fixed, 0.0], dtype=jnp.float64)
        if verbose:
            print(f"[MV-FIX] Prescribing toroidal flux Φ_t={toroidal_flux:.6g} ⇒ a_t={a_t_fixed:.6g}; setting a_p=0.")
        # Rebuild g_raw with fixed a
        grad_t_bdry = grad_t(Pn)
        grad_p_bdry = grad_p(Pn)
        g_raw = scinfo.scale * jnp.sum(Nn * (a[0]*grad_t_bdry + a[1]*grad_p_bdry), axis=1)

    # === Prepare constraints ===
    # Boundary constraint:
    c_bdry, d_bdry = make_flux_constraint(A, W, g_raw)

    # Interior ring for *constraint* (same ring you use for diagnostics)
    eps_n = max(1e-6, interior_eps_factor * h_med)    # normalized
    eps_w = eps_n / scinfo.scale
    P_in  = P - eps_w * N
    Xn_in = (P_in - scinfo.center) * scinfo.scale
    A_in  = build_A_rows_at_points(Xn_in, N, Yn, scinfo)

    # g_in from MV on interior:
    # OPTIONAL speed path: reuse boundary normals for ring; ϕ̂_tan, θ̂ remain consistent for tiny offsets
    use_fast_ring = (kind == "torus")  # mirrors need accurate θ̂
    Gt_in = grad_t(Xn_in)
    if use_fast_ring:
        # Build poloidal direction with boundary normals to avoid nearest-neighbor search
        x, y = Xn_in[:, 0], Xn_in[:, 1]
        r2   = jnp.maximum(1e-30, x*x + y*y)
        phi_hat   = jnp.stack([-y / jnp.sqrt(r2), x / jnp.sqrt(r2), jnp.zeros_like(x)], axis=1)
        phi_tan   = phi_hat - jnp.sum(phi_hat * N, axis=1, keepdims=True) * N
        phi_tan   = phi_tan / jnp.maximum(1e-30, jnp.linalg.norm(phi_tan, axis=1, keepdims=True))
        theta_hat = jnp.cross(N, phi_tan)
        theta_hat = theta_hat / jnp.maximum(1e-30, jnp.linalg.norm(theta_hat, axis=1, keepdims=True))
        Gp_in = theta_hat
    else:
        Gp_in = grad_p(Xn_in)

    g_in  = scinfo.scale * jnp.sum(N * (a[0]*Gt_in + a[1]*Gp_in), axis=1)

    # Build ring weights once and use them for the constraint
    W_ring = build_ring_weights(P_in, Pn, k=k_nn)
    c_int, d_int = make_flux_constraint(A_in, W_ring, g_in)

    # --- Add two virtual end-cap flux constraints (close the open sleeve) ---
    try:
        c_cap_low,  d_cap_low  = build_cap_flux_constraint(P, N, Pn, Nn, scinfo, Yn,
                                                        grad_t, grad_p, a, best["a_hat"],
                                                        q=0.02, ds_frac=0.02, k_cap=k_nn, side="low")
        c_cap_high, d_cap_high = build_cap_flux_constraint(P, N, Pn, Nn, scinfo, Yn,
                                                        grad_t, grad_p, a, best["a_hat"],
                                                        q=0.02, ds_frac=0.02, k_cap=k_nn, side="high")
        cap_constraints = [(c_cap_low,  -d_cap_low), (c_cap_high, -d_cap_high)]
    except Exception as e:
        print("[WARN] Cap constraints failed; continuing without them:", e)
        cap_constraints = []

    # Choose your policy:
    # 1) Most robust in practice: enforce on Γ₋ only
    constraints = [(c_bdry, -d_bdry)]
    # 2) Enforce on both Γ and Γ₋ (two constraints):
    # constraints = [(c_bdry, -d_bdry), (c_int, -d_int)]
    # 3) Enforce on Γ, Γ₋, and end-caps (four constraints):
    # constraints = [(c_bdry, -d_bdry), (c_int, -d_int)] + cap_constraints

    alpha = augmented_lagrangian_solve(
        A, W, g_raw, lam=lambda_reg_opt, constraints=constraints,
        rho0=1e2, rho_max=1e4, iters=6, verbose=verbose
    )
    
    print(f"[SOL] a_t={float(a[0]):.6g}, a_p={float(a[1]):.6g}, ||alpha||₂={float(jnp.linalg.norm(alpha)):.3e}")

    phi_fn, grad_fn, psi_fn, grad_psi_fn, lap_psi_fn, _ = build_evaluators_mfs(
        Pn, Yn, alpha, phi_t, phi_p, a, scinfo, grad_t, grad_p
    )

    # === Diagnostics on Γ (illustrative; gradients may still be large due to proximity to sources) ===
    grad_on_Gamma = grad_fn(P)
    vec_stats("[EVAL Γ] |∇φ|", jnp.linalg.norm(grad_on_Gamma, axis=1))

    # === Diagnostics on Γ₋ (interior offset): reliable ===
    eps_n = max(1e-6, interior_eps_factor * h_med)  # normalized offset inward
    eps_w = eps_n / scinfo.scale                     # world offset
    P_in = P - eps_w * N
    print(f"[DIAG] Using interior-offset ring Γ₋ with eps_n={eps_n:.3g} (normalized), eps_world={eps_w:.3g}.")

    grad_on_ring = grad_fn(P_in)
    grad_mag_ring = jnp.linalg.norm(grad_on_ring, axis=1)
    vec_stats("[EVAL Γ₋] |∇φ|", grad_mag_ring)
    
    n_dot_grad_ring = jnp.sum(N * grad_on_ring, axis=1)
    grad_mag_ring   = jnp.linalg.norm(grad_on_ring, axis=1)
    rn = jnp.abs(n_dot_grad_ring) / jnp.maximum(1e-30, grad_mag_ring)
    vec_stats("[EVAL Γ₋] normalized BC |n·∇φ|/|∇φ|", rn)

    # Flux neutrality on Γ₋ (should be ~0)
    # reuse the ring weights from the constraint stage
    n_dot_grad_ring = jnp.sum(N * grad_on_ring, axis=1)
    flux = float(jnp.dot(W_ring, n_dot_grad_ring))
    area_ring = float(jnp.sum(W_ring))
    print(f"[CHK] Flux neutrality on Γ₋: ∫ n·∇φ dS ≈ {flux:.6e}  (avg={flux/area_ring:.3e})")

    def grad_cyl_about_axis(P, a_hat):
        # ∇ϕ_a = (a × r_perp)/|r_perp|^2 with r_perp = (X - c) - ((X - c)·a)a
        a = a_hat / np.linalg.norm(a_hat)
        X = np.asarray(P)
        c = np.asarray(scinfo.center)          # <<<<<< use the same center
        Xc = X - c
        r_par  = (Xc @ a)[:,None] * a[None,:]
        r_perp = Xc - r_par
        r2     = np.sum(r_perp*r_perp, axis=1, keepdims=True)
        return np.cross(a[None,:], r_perp) / np.maximum(1e-30, r2)

    Gt = np.asarray(grad_t(Pn))          # normalized-space grad_t
    Gc = grad_cyl_about_axis(P, np.array(best["a_hat"]))  # world-space, but only direction matters
    # compare directions only
    c = np.sum(Gt*Gc, axis=1)/(np.linalg.norm(Gt,axis=1)*np.linalg.norm(Gc,axis=1)+1e-30)
    print("median cos(angle(grad_t, axis-aware cylindrical)) ≈", np.median(c))

    # Laplacian(ψ) near boundary (independent check)
    lap_in = lap_psi_fn(P_in)
    vec_stats("[LAP Γ₋] |∇²ψ|", jnp.abs(lap_in))

    # Plots (all on Γ₋)
    plot_geometry_and_solution(P, N, grad_on_ring, title_suffix="",
                           show_normals=show_normals, kind=kind, a_hat=best.get("a_hat", None))
    plot_boundary_condition_errors(P, N, grad_on_ring)
    plot_laplacian_errors_on_interior_band(P, lap_in, eps_w)

    return dict(
        P=P, N=N, Pn=Pn, W=W, rk=rk, h_med=h_med,
        alpha=alpha, a=a, delta_n=delta_n, eps_n=eps_n,
        phi_fn=phi_fn, grad_fn=grad_fn, psi_fn=psi_fn,
        grad_psi_fn=grad_psi_fn, laplacian_psi_fn=lap_psi_fn,
        scinfo=scinfo, Yn=Yn, a_hat=best["a_hat"], kind=kind
    )

if __name__ == "__main__":
    candidate_xyz, candidate_normals = get_candidates(script_dir, file_name, subdir="inputs")
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz", nargs="?", default=candidate_xyz,
                    help="CSV file with x,y,z columns (positional or --xyz)")
    ap.add_argument("normals", nargs="?", default=candidate_normals,
                    help="CSV file with nx,ny,nz columns (positional or --nrm)")
    ap.add_argument("--sf_min", type=float, default=1.0, help="Min source factor for autotuning")
    ap.add_argument("--sf_max", type=float, default=6.5, help="Max source factor for autotuning")
    ap.add_argument("--lbfgs-maxiter", type=int, default=10, help="Max iterations for L-BFGS")
    ap.add_argument("--lbfgs-tol", type=float, default=1e-8, help="Tolerance for L-BFGS")
    ap.add_argument("--k-nn", type=int, default=26, help="k for kNN geometry stats")
    ap.add_argument("--no-use-mv", dest="use_mv", action="store_false", help="Disable MV regularization")
    ap.add_argument("--no-verbose", dest="verbose", action="store_false", help="Disable verbose logging")
    ap.add_argument("--mv-weight", type=float, default=0.5)
    ap.add_argument("--interior-eps-factor", type=float, default=5e-3)
    ap.add_argument("--show-normals", action="store_true")
    ap.add_argument("--toroidal-flux", type=float, default=None,
                    help="If set, prescribes the toroidal flux Φ_t (sets a_t = Φ_t/(2π))")
    ap.add_argument("--mfs-out", default=None,
                    help="Write portable MFS solution to this .npz (center,scale,Yn,alpha,a,a_hat,P,N,kind)")
    ap.add_argument("--geom-kind", choices=["auto", "torus", "mirror"], default="auto",
                    help="Override geometry classification (torus/mirror) or let it auto-detect")
    ap.add_argument("--bc-weight", type=float, default=1e5,
                    help="Weight for boundary condition constraint in autotuning")
    args = ap.parse_args()

    if "knot" in os.path.basename(args.xyz).lower():
        print("[INFO] Detected knot-like surface; enabling knotatron defaults:")
        # Keep torus-like classification (good for a tube), but DO NOT disable MV:
        # print("       - Disabling multivalued basis (use_mv=False)")
        print("       - Forcing torus-like geometry classification")
        # leave args.use_mv unchanged so you can still pass --no-use-mv if you really want
        if args.geom_kind == "auto":
            args.geom_kind = "torus"

    print(args.xyz, args.normals)
    if args.mfs_out == None:
        args.mfs_out = args.xyz.replace(".csv", "_solution.npz")
        args.mfs_out = str((script_dir / ".." / "outputs" / os.path.basename(args.mfs_out)).resolve())
        print(f"[INFO] No --mfs-out provided; defaulting to {args.mfs_out}")

    out = main(
        xyz_csv=args.xyz, nrm_csv=args.normals,
        use_mv=args.use_mv, k_nn=args.k_nn,
        mv_weight=args.mv_weight,
        interior_eps_factor=args.interior_eps_factor,
        verbose=args.verbose,
        show_normals=args.show_normals,
        toroidal_flux=args.toroidal_flux,
        sf_min=args.sf_min, sf_max=args.sf_max,
        lbfgs_maxiter=args.lbfgs_maxiter, lbfgs_tol=args.lbfgs_tol,
        geom_kind=args.geom_kind, bc_weight=args.bc_weight
    )

    # --- SAVE a portable checkpoint for the tracer ---
    try:
        np.savez(
            args.mfs_out,
            center=np.asarray(out["scinfo"].center, dtype=float),
            scale=float(np.asarray(out["scinfo"].scale)),
            Yn=np.asarray(out["Yn"], dtype=float),
            alpha=np.asarray(out["alpha"], dtype=float),
            a=np.asarray(out["a"], dtype=float),
            a_hat=np.asarray(out["a_hat"], dtype=float),
            P=np.asarray(out["P"], dtype=float),
            N=np.asarray(out["N"], dtype=float),
            kind=str(out["kind"]),
        )
        print(f"[SAVE] Wrote MFS checkpoint → {args.mfs_out}")
    except Exception as e:
        print("[WARN] Could not save MFS checkpoint:", e)
