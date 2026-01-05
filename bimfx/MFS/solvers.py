import jax.numpy as jnp
from jax import debug as jax_debug
from jax import lax, jit, value_and_grad, vmap
import numpy as np
from sklearn.neighbors import NearestNeighbors
from functools import partial
import jax.scipy.linalg as jsp_linalg

from bimfx.MFS.geometry import (best_fit_axis, project_to_local, detect_geometry_and_axis,
                                multivalued_bases_about_axis, orthonormal_complement)
from bimfx.MFS.sources_kernels import build_mfs_sources, build_evaluators_mfs, grad_green_x
from bimfx.utils.printing import vec_stats

# ----------------------------- System build ---------------------- #
@partial(jit,static_argnames=("grad_t", "grad_p", "use_mv", "center_D", "verbose"))
def build_system_matrices(Pn, Nn, Yn, W, grad_t, grad_p, scinfo, use_mv=True, center_D=True, verbose=True):
    """
    Build collocation matrix for Neumann BC:
      A_ij = n_i · ∇_x G( x_i , y_j ),  x_i = Pn[i], y_j = Yn[j]
    Returns A (world units). D is not needed in the current solve path.
    """
    X = Pn

    @jit
    def row_kernel(xi, ni):
        grads = vmap(lambda yj: grad_green_x(xi, yj))(Yn)   # (M,3)
        return jnp.dot(grads, ni)                           # (M,)

    A = vmap(row_kernel)(X, Nn)                              # (N, M)
    A = scinfo.scale * A

    if verbose:
        # All JAX ops; no numpy, no Python formatting of tracers
        shape0 = A.shape[0]; shape1 = A.shape[1]
        Amin = jnp.min(jnp.abs(A)); Amed = jnp.median(jnp.abs(A)); Amax = jnp.max(jnp.abs(A))
        jax_debug.print(
            "[SYS] A shape=({n},{m}), |A| stats: min={mn:.3e}, median={md:.3e}, max={mx:.3e}",
            n=shape0, m=shape1, mn=Amin, md=Amed, mx=Amax)
    # keep old signature but return dummy D to avoid changing callers
    D = jnp.zeros((Pn.shape[0], 2), dtype=jnp.float64)
    return A, D

# ----------------------- 2D quasi-Newton (BFGS) ----------------------- #
def bfgs_2d(value_and_grad_fn, p0, *,
            sf_min=1.0, sf_max=4.0,
            max_iter=25, tol=1e-6):
    """
    Tiny 2D BFGS in JAX over p=[log_sf, log_lambda] with early stopping.
    Early stop when ||g|| < tol or iterations reach max_iter.
    """
    def project(p):
        # p = [log_sf, log_lam]
        log_sf, log_lam = p
        sf = jnp.exp(log_sf)
        sf_clamped = jnp.clip(sf, sf_min, sf_max)
        log_lam_clamped = jnp.clip(log_lam, -9.0, -4.0)  # λ in [e^-9, e^-3] ≈ [1e-4, 5e-2]
        return jnp.array([jnp.log(sf_clamped), log_lam_clamped], dtype=p.dtype)

    def one_step(state):
        # state = (p, H, f_prev, it, done)
        p, H, f_prev, it, done = state

        # If already done, just return the same state (no-ops) to keep shapes static
        def _advance(_):
            p_proj = project(p)
            f, g = value_and_grad_fn(p_proj)

            jax_debug.print("[BFGS] it={it}  f={f:.3e}  ||g||={gn:.2e}  sf={sf:.3f}  lam={lam:.3e}",
                            it=it, f=f, gn=jnp.linalg.norm(g),
                            sf=jnp.exp(p_proj[0]), lam=jnp.exp(p_proj[1]))

            # descent direction
            d = - H @ g

            # Backtracking line search (Armijo)
            def bt_body(carry, _):
                step, f_curr = carry
                p_try = project(p_proj + step * d)
                f_try, _ = value_and_grad_fn(p_try)
                ok = f_try <= f + 1e-4 * step * (g @ d)
                step = jnp.where(ok, step, 0.5 * step)
                f_curr = jnp.where(ok, f_try, f_curr)
                return (step, f_curr), ok

            (step_final, f_new), _ = lax.scan(
                bt_body,
                (jnp.array(1.0, dtype=p.dtype), f),
                jnp.arange(4)   # up to 4 backtracks
            )
            p_new = project(p_proj + step_final * d)

            # BFGS update with mild Powell damping to keep H SPD
            _, g_new = value_and_grad_fn(p_new)
            s = p_new - p_proj
            y = g_new - g
            ys = y @ s
            # Powell damping: ensure y^T s is not too small
            theta = jnp.where(ys < 0.2 * (s @ (H @ s)), 
                            (0.8 * (s @ (H @ s))) / (s @ (H @ s) - ys + 1e-30),
                            1.0)
            y_tilde = theta * y + (1 - theta) * (H @ s)
            rho = 1.0 / (y_tilde @ s + 1e-30)
            I = jnp.eye(2, dtype=p.dtype)
            V = I - rho * jnp.outer(s, y_tilde)
            H_new = V @ H @ V.T + rho * jnp.outer(s, s)

            # early stop: gradient small OR relative f change small
            rel = jnp.abs((f_new - f) / (jnp.abs(f) + 1e-30))
            done_new = jnp.logical_or(jnp.linalg.norm(g_new) < tol, rel < 1e-3)
            return (p_new, H_new, f_new, it + 1, done_new)

        def _passthrough(_):
            return (p, H, f_prev, it + 1, done)

        # If done, do passthrough; else, advance one BFGS step
        return lax.cond(done, _passthrough, _advance, operand=None)

    # Initial state
    H0 = jnp.eye(2, dtype=p0.dtype)
    f0 = jnp.inf
    state0 = (p0, H0, f0, jnp.array(0, dtype=jnp.int32), jnp.array(False))

    def cond_fun(state):
        _, _, f_prev, it, done = state
        # stop if done or max_iter; also stop if relative change in f is tiny
        return jnp.logical_and(it < max_iter, jnp.logical_not(done))

    p_star, H_star, f_star, _, _ = lax.while_loop(cond_fun, one_step, state0)
    f_star, g_star = value_and_grad_fn(project(p_star))
    return p_star, f_star, g_star, H_star


def build_ring_weights(P_in, Pn, k=32):
    """
    kNN area weights W_ring on the interior ring Γ₋.
    Reuses the best-fit (u,v)-plane from the boundary Pn to define distances.
    """
    # Plane from boundary (stays constant for the surface)
    c_plane, E_plane, _ = best_fit_axis(np.array(Pn), verbose=False)
    # Project ring to that plane
    Ploc_ring = project_to_local(P_in, c_plane, E_plane)
    XY = np.asarray(Ploc_ring[:, :2])

    k_eff = min(k+1, len(XY))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(XY)
    dists, _ = nbrs.kneighbors(XY)
    rk_ring = dists[:, -1]  # k-th neighbor radius
    W_ring = jnp.asarray(np.pi * rk_ring**2, dtype=jnp.float64)
    return W_ring

def solve_once(P, N, Pn, Nn, W, rk, scinfo,
               use_mv, k_nn, source_factor, lambda_reg, mv_weight,
               interior_eps_factor, verbose=True):
    # Build sources and system
    Yn, delta_n = build_mfs_sources(Pn, Nn, rk, scinfo, source_factor=source_factor, verbose=verbose)
    # sanity: ensure δ points outward (dot > 0 for most points)
    dot = jnp.sum((Yn - Pn) * Nn, axis=1)
    print(f"[CHK] (Yn - Pn)·Nn: min={float(dot.min()):.3e}, median={float(jnp.median(dot)):.3e}")
    kind, a_hat, E_axes, c_axes, svals = detect_geometry_and_axis(Pn, verbose=True)
    phi_t, grad_t, phi_p, grad_p = multivalued_bases_about_axis(Pn, Nn, a_hat, verbose=True) if use_mv else (
        (lambda Xn: jnp.zeros((Xn.shape[0],))), (lambda Xn: jnp.zeros_like(Xn)),
        (lambda Xn: jnp.zeros((Xn.shape[0],))), (lambda Xn: jnp.zeros_like(Xn))
    )
    A, D = build_system_matrices(Pn, Nn, Yn, W, grad_t, grad_p, scinfo,
                                 use_mv=use_mv, center_D=True, verbose=verbose)

    # Fit multivalued coefficients a and rhs
    if use_mv:
        grad_t_bdry, grad_p_bdry = grad_t(Pn), grad_p(Pn)
        a, D_raw, D0 = fit_mv_coeffs_minimize_rhs(Nn, W, grad_t_bdry, grad_p_bdry, verbose=verbose)
        g_raw = jnp.sum(Nn * (a[0]*grad_t_bdry + a[1]*grad_p_bdry), axis=1)
    else:
        a = jnp.zeros((2,), dtype=jnp.float64)
        g_raw = jnp.zeros((Pn.shape[0],), dtype=jnp.float64)

    # Solve for alpha
    alpha = solve_alpha_with_rhs(A, W, g_raw, lam=lambda_reg, verbose=verbose)

    # Build evaluators
    phi_fn, grad_fn, psi_fn, grad_psi_fn, lap_psi_fn, _ = build_evaluators_mfs(
        Pn, Yn, alpha, phi_t, phi_p, a, scinfo, grad_t, grad_p
    )

    # Interior ring diagnostics
    Wsum = float(jnp.sum(W))
    Wsqrt = jnp.sqrt(W)
    h_med = float(np.median(rk))
    eps_n = max(1e-6, interior_eps_factor * h_med)
    eps_w = eps_n / scinfo.scale
    P_in  = P - eps_w * N
    grad_on_ring = grad_fn(P_in)
    n_dot_grad   = jnp.sum(N * grad_on_ring, axis=1)
    bc_w2 = float(jnp.sqrt(jnp.dot(W, n_dot_grad**2)))          # ||n·∇φ||_W2 on Γ₋
    flux = float(jnp.dot(W, n_dot_grad))                        # ∫Γ₋ n·∇φ dS
    grad_mag_ring = jnp.linalg.norm(grad_on_ring, axis=1)
    lap_in = lap_psi_fn(P_in)
    lap_l2 = float(jnp.linalg.norm(lap_in))                     # ||∇²ψ||_2 on Γ₋
    alpha_norm = float(jnp.linalg.norm(alpha))

    # Rough normal-equations condition proxy (reuse LS-α build)
    Aw = Wsqrt[:, None] * A
    ATA = Aw.T @ Aw
    NE  = ATA + (lambda_reg**2) * jnp.eye(A.shape[1], dtype=A.dtype)
    condNE = float(np.linalg.cond(np.asarray(NE)))

    metrics = dict(
        bc_w2=bc_w2,
        flux_abs=abs(flux),
        lap_l2=lap_l2,
        alpha_norm=alpha_norm,
        condNE=condNE,
        a=a,
        delta_n=delta_n,
        eps_n=eps_n,
        eps_w=eps_w,
        grad_on_ring=grad_on_ring,
        lap_in=lap_in,
        phi_fn=phi_fn,
        grad_fn=grad_fn
    )
    return alpha, metrics

def autotune(P, N, Pn, Nn, W, rk, scinfo,
             use_mv=True,
             mv_weight=0.5,
             interior_eps_factor=5e-3,
             verbose=True, bc_weight=20.0,
             sf_min=1.0, sf_max=6.5,
             lbfgs_maxiter=15, lbfgs_tol=1e-8,
             geom_kind="auto"):
    if geom_kind == "auto":
        kind, a_hat, E_axes, c_axes, svals = detect_geometry_and_axis(Pn, verbose=True)
    else:
        # Force the 'kind' but still get some axis from PCA
        auto_kind, a_hat, E_axes, c_axes, svals = detect_geometry_and_axis(Pn, verbose=True)
        kind = geom_kind

    # --- Multivalued bases + initial a, g_raw ---
    if use_mv:
        # true MV bases
        phi_t, grad_t, phi_p, grad_p = multivalued_bases_about_axis(Pn, Nn, a_hat, verbose=True)
        gt_b, gp_b = grad_t(Pn), grad_p(Pn)
        a, _, _ = fit_mv_coeffs_minimize_rhs(Nn, W, gt_b, gp_b, verbose=False)
        g_raw = scinfo.scale * jnp.sum(Nn * (a[0]*gt_b + a[1]*gp_b), axis=1)
    else:
        # turn MV completely off
        def phi_t(Xn): return jnp.zeros((Xn.shape[0],), dtype=jnp.float64)
        def grad_t(Xn): return jnp.zeros_like(Xn)
        def phi_p(Xn): return jnp.zeros((Xn.shape[0],), dtype=jnp.float64)
        def grad_p(Xn): return jnp.zeros_like(Xn)
        a = jnp.zeros((2,), dtype=jnp.float64)
        g_raw = jnp.zeros((Pn.shape[0],), dtype=jnp.float64)

    # ---- 2D optimization over (log_sf, log_lambda) ----
    h_med = float(np.median(rk))
    obj = make_objective_for_delta_lambda(
        P, N, Pn, Nn, W, rk, scinfo,
        grad_t, grad_p, a, g_raw,
        interior_eps_factor=interior_eps_factor, h_med=h_med,
        use_mv=use_mv, bc_weight=bc_weight,
    )

    # robust starting point (you were scanning 1.5..2.5)
    log_sf0  = jnp.log(1.5)
    log_lam0 = jnp.log(1e-4)
    p0 = jnp.array([log_sf0, log_lam0], dtype=jnp.float64)

    # small box for sf, keep λ unbounded in log-space (still positive)
    p_star, f_star, g_star, H_star = bfgs_2d(obj, p0, sf_min=sf_min, sf_max=sf_max, max_iter=lbfgs_maxiter, tol=lbfgs_tol)
    log_sf_star, log_lam_star = p_star
    sf_star  = float(jnp.exp(log_sf_star))
    lam_star = float(jnp.exp(log_lam_star))

    # Build final sources at optimum and reuse downstream
    Yn, _ = build_mfs_sources(Pn, Nn, rk, scinfo, source_factor=sf_star, verbose=False)

    if verbose:
        print(f"[OPT] δ/source_factor* = {sf_star:.3f}, λ* = {lam_star:.3e}, J* = {float(f_star):.3e}")

    return dict(
        source_factor=sf_star, lambda_reg=lam_star,
        a=a, phi_t=phi_t, grad_t=grad_t, phi_p=phi_p, grad_p=grad_p, Yn=Yn,
        a_hat=a_hat, kind=kind
    )

def make_objective_for_delta_lambda(P, N, Pn, Nn, W, rk, scinfo,
                                    grad_t, grad_p, a, g_raw,
                                    interior_eps_factor, h_med, use_mv,
                                    bc_weight=10.0):
    eps_factor_obj = jnp.maximum(2e-2, interior_eps_factor)
    Wsqrt = jnp.sqrt(W)
    eps_n = jnp.maximum(1e-6, eps_factor_obj * h_med)
    eps_w = eps_n / scinfo.scale
    P_in  = P - eps_w * N

    # --- NEW: precompute ring weights for the objective (constant wrt p) ---
    # Use a modest k to keep it cheap; it does not depend on (sf, λ).
    W_ring_obj = build_ring_weights(P_in, Pn, k=32)

    def objective(p):
        log_sf, log_lam = p
        sf  = jnp.exp(log_sf)
        lam = jnp.exp(log_lam)

        Yn, _ = build_mfs_sources(Pn, Nn, rk, scinfo, source_factor=sf, verbose=False)
        A, _  = build_system_matrices(Pn, Nn, Yn, W, grad_t, grad_p, scinfo,
                                      use_mv=use_mv, center_D=True, verbose=False)

        # Boundary quantities (unchanged)
        c_bdry, d_bdry = make_flux_constraint(A, W, g_raw)

        # Interior-ring constraint — use the precomputed ring weights
        Xn_in = (P_in - scinfo.center) * scinfo.scale
        A_in  = build_A_rows_at_points(Xn_in, N, Yn, scinfo)
        Gt_in = grad_t(Xn_in); Gp_in = grad_p(Xn_in)
        g_in  = scinfo.scale * jnp.sum(N * (a[0]*Gt_in + a[1]*Gp_in), axis=1)
        c_int, d_int = make_flux_constraint(A_in, W_ring_obj, g_in)

        alpha = solve_alpha_with_rhs_hard_flux_multi(
            A, W, g_raw, lam=lam, constraints=[(c_int, -d_int,), (c_bdry, -d_bdry)],
            verbose=False
        )

        res_w = Wsqrt * (A @ alpha + g_raw)
        term_res = jnp.dot(res_w, res_w)

        zero_like = lambda Xn: jnp.zeros((Xn.shape[0],), dtype=Xn.dtype)
        phi_t0, grad_t_fn, phi_p0, grad_p_fn = zero_like, grad_t, zero_like, grad_p
        phi_fn, grad_fn, _, _, _, _ = build_evaluators_mfs(
            Pn, Yn, alpha, phi_t0, phi_p0, a, scinfo, grad_t_fn, grad_p_fn
        )
        grad_in = grad_fn(P_in)
        n_dot   = jnp.sum(N * grad_in, axis=1)
        term_bc = jnp.dot(W, n_dot**2)

        reg = 1e-6 * (log_sf**2 + log_lam**2)
        return term_res + bc_weight * term_bc + reg

    return jit(value_and_grad(objective))


# ----------------------- Regularized weighted LS ------------------------ #
def solve_alpha_with_rhs(A, W, g_raw, lam=1e-3, verbose=True):
    """ Solve min || W^{1/2}(A α + g_raw)||_2^2 + λ^2 ||α||_2^2 for α only. """
    Wsqrt = jnp.sqrt(W)
    Aw = Wsqrt[:, None] * A
    gw = Wsqrt * g_raw
    ATA = Aw.T @ Aw
    ATg = Aw.T @ gw
    NE = ATA + (lam**2) * jnp.eye(A.shape[1], dtype=A.dtype)
    rhs = -ATg
    condNE = np.linalg.cond(np.asarray(NE))
    if verbose:
        print(f"[LS-α] size={NE.shape}, cond≈{condNE:.3e}, λ={lam:.3e}")
    L = jnp.linalg.cholesky(NE)
    y = jsp_linalg.solve_triangular(L, rhs, lower=True)
    alpha = jsp_linalg.solve_triangular(L.T, y, lower=False)
    res = Aw @ alpha + gw
    if verbose:
        vec_stats("[LS-α] weighted residual", res)
    return alpha

def solve_alpha_with_rhs_hard_flux_multi(A, W, g_raw, lam=1e-3, constraints=None, verbose=True):
    """
    Solve min ||W^{1/2}(A α + g)||^2 + λ^2||α||^2  s.t. C^T α = d
    using Schur complement: μ from (C^T NE^{-1} C)μ = d - C^T NE^{-1} rhs1,
    then α = NE^{-1}(rhs1 - C μ), where NE = A^T W A + λ^2 I, rhs1 = -A^T W g.
    """
    if not constraints:
        return solve_alpha_with_rhs(A, W, g_raw, lam=lam, verbose=verbose)

    Wv  = jnp.asarray(W).reshape(-1)
    ATW = A.T * Wv[None, :]
    NE  = ATW @ A + (lam**2) * jnp.eye(A.shape[1], dtype=A.dtype)
    rhs1 = - (ATW @ g_raw)

    # Build C (M x K) and d (K,)
    Ccols, dlist = [], []
    for (c_vec, d_k) in constraints:
        Ccols.append(c_vec[:, None])
        dlist.append(d_k)
    C = jnp.concatenate(Ccols, axis=1)             # (M,K)
    d = jnp.asarray(dlist, dtype=NE.dtype)         # (K,)

    # Cholesky of NE
    L = jnp.linalg.cholesky(NE)

    # Solve NE^{-1} rhs1 and NE^{-1} C (columns)
    y1 = jsp_linalg.solve_triangular(L, rhs1, lower=True)
    z1 = jsp_linalg.solve_triangular(L.T, y1, lower=False)      # z1 = NE^{-1} rhs1

    Y  = jsp_linalg.solve_triangular(L, C, lower=True)
    Z  = jsp_linalg.solve_triangular(L.T, Y, lower=False)       # Z = NE^{-1} C

    # Schur system: S μ = d - C^T z1,  where S = C^T Z = C^T NE^{-1} C
    S   = C.T @ Z                                               # (K,K)
    rhsμ = d - (C.T @ z1)
    μ   = jnp.linalg.solve(S, rhsμ)

    # α = NE^{-1}(rhs1 - C μ) = z1 - Z μ
    alpha = z1 - Z @ μ

    if verbose:
        res   = A @ alpha + g_raw
        lw2   = float(jnp.sqrt(jnp.dot(Wv, res**2)))
        condNE = float(np.linalg.cond(np.asarray(NE)))
        condS  = float(np.linalg.cond(np.asarray(S)))
        print(f"[LS-α:Schur] NE cond≈{condNE:.3e}, S cond≈{condS:.3e}, λ={lam:.3g}, ||W^0.5 res||={lw2:.3e}")
    return alpha

def augmented_lagrangian_solve(A, W, g_raw, lam, constraints, rho0=1e-2, rho_max=1e3, iters=5, verbose=True):
    """
    Augmented Lagrangian on linear constraints C^T alpha = d:
      minimize ||W^{1/2}(A alpha + g)||^2 + lam^2||alpha||^2
              + μ^T(C^T alpha - d) + (ρ/2)||C^T alpha - d||^2
    We eliminate μ analytically by folding into the RHS with iterative μ-updates.
    """
    # pack constraints -> matrix C (M x K) and vector d (K,)
    Ccols, dlist = [], []
    for (c_vec, d_k) in constraints:
        Ccols.append(c_vec[:, None])
        dlist.append(d_k)
    C = jnp.concatenate(Ccols, axis=1)  # (M,K)
    d = jnp.asarray(dlist, dtype=A.dtype)

    Wv  = jnp.asarray(W).reshape(-1)
    ATW = A.T * Wv[None, :]
    ATA = ATW @ A
    ATg = ATW @ g_raw

    # initialize
    K = C.shape[1]
    mu = jnp.zeros((K,), dtype=A.dtype)
    rho = rho0
    alpha = jnp.zeros((A.shape[1],), dtype=A.dtype)

    for it in range(iters):
        # NE(ρ) := A^T W A + lam^2 I + ρ C C^T  (note: C C^T is (M x M), we need in M-space)
        # Work in α-space via Schur: form NE = ATA + lam^2 I + ρ * (C @ C.T) projected via α
        # More efficiently: normal eq. with extra term via (C (C^T α)) handled as (ρ C C^T) in M-space:
        # We augment RHS to include μ and d: rhs = -(ATg) - C (μ + ρ * d)
        rhs = -ATg - C @ (mu + rho * d)

        NE = ATA + (lam**2) * jnp.eye(A.shape[1], dtype=A.dtype) + rho * (C @ C.T)

        # Cholesky solve
        L = jnp.linalg.cholesky(NE)
        y = jsp_linalg.solve_triangular(L, rhs, lower=True)
        alpha = jsp_linalg.solve_triangular(L.T, y, lower=False)

        # constraint residual and multipliers update
        r = (C.T @ alpha) - d
        mu = mu + rho * r

        if verbose:
            lw2 = float(jnp.sqrt(jnp.dot(Wv, (A @ alpha + g_raw)**2)))
            crel = float(jnp.linalg.norm(r) / (jnp.linalg.norm(d) + 1e-30))
            print(f"[AL] it={it}  ||W^1/2 res||={lw2:.3e}  ||C^Tα-d||/||d||={crel:.3e}  rho={rho:.2e}")

        rho = min(rho * 10.0, rho_max)

    return alpha

@jit
def build_A_rows_at_points(Xn_eval, N_eval, Yn, scinfo):
    @jit
    def row_kernel(xi, ni):
        grads = vmap(lambda yj: grad_green_x(xi, yj))(Yn)   # (M,3)
        return jnp.dot(grads, ni)                           # (M,)
    A_in = vmap(row_kernel)(Xn_eval, N_eval)                # (N,M)
    return scinfo.scale * A_in                              # to world units



def build_cap_flux_constraint(P, N, Pn, Nn, scinfo, Yn,
                              grad_t, grad_p, a, a_hat,
                              q=0.02, ds_frac=0.02, k_cap=32, side="low"):
    """
    Build (c_cap, d_cap) for a virtual end-cap perpendicular to a_hat.
    - side: "low" or "high": chooses s-quantile q or 1-q
    - ds_frac: half-thickness in axial coordinate as a fraction of axial span
    """
    X = jnp.asarray(P)
    a = jnp.asarray(a_hat) / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    c = jnp.asarray(scinfo.center)

    # axial coordinate s and span
    s = jnp.sum((X - c[None,:]) * a[None,:], axis=1)
    s_min, s_max = float(jnp.min(s)), float(jnp.max(s))
    s_span = max(1e-9, s_max - s_min)

    s0 = np.quantile(np.asarray(s), q if side == "low" else 1.0 - q)
    ds = ds_frac * s_span
    mask = np.abs(np.asarray(s) - s0) <= ds
    if not np.any(mask):
        raise RuntimeError(f"No points found for {side} cap; try increasing ds_frac.")

    # cap points and normals (constant ±a_hat)
    P_cap = X[mask, :]
    N_cap = ( -a if side == "low" else a )[None, :].repeat(P_cap.shape[0], axis=0)

    # weights on cap in plane ⟂ a_hat via kNN
    # build plane basis (e1,e2) ⟂ a_hat
    e1_np, e2_np = orthonormal_complement(np.array(a_hat))
    e1 = jnp.asarray(e1_np); e2 = jnp.asarray(e2_np)
    Xc = P_cap - c[None, :]
    u1 = np.asarray(jnp.sum(Xc * e1[None,:], axis=1))
    u2 = np.asarray(jnp.sum(Xc * e2[None,:], axis=1))
    UV = np.column_stack([u1, u2])

    k_eff = min(k_cap+1, len(UV))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(UV)
    dists, _ = nbrs.kneighbors(UV)
    rk_cap = dists[:, -1]
    W_cap = jnp.asarray(np.pi * rk_cap**2, dtype=jnp.float64)

    # collocation on the cap (use normalized coords for grad_t/grad_p)
    Xn_cap = (P_cap - scinfo.center) * scinfo.scale
    A_cap = build_A_rows_at_points(Xn_cap, N_cap, Yn, scinfo)

    # MV contribution on the cap
    Gt_cap = grad_t(Xn_cap)
    Gp_cap = grad_p(Xn_cap)
    g_cap  = scinfo.scale * jnp.sum(N_cap * (a[0]*Gt_cap + a[1]*Gp_cap), axis=1)

    # flux constraint vector for the cap
    c_cap, d_cap = make_flux_constraint(A_cap, W_cap, g_cap)
    return c_cap, d_cap


def make_flux_constraint(A_like, W_like, g_like):
    # Return (c_vec, d_scalar) with c_vec = A_like^T W_like 1, d = W_like·g_like
    Wv = jnp.asarray(W_like).reshape(-1)
    c_vec = (A_like.T * Wv[None, :]).sum(axis=1)   # (M,)
    d_val = jnp.dot(Wv, g_like)                    # scalar
    return c_vec, d_val

# ------------------ Multivalued bases (2D) ---------------------- #
def fit_mv_coeffs_minimize_rhs(Nn, W, grad_t_bdry, grad_p_bdry, verbose=True):
    """
    Weighted LS *directional* fit for a=(a_t,a_p), using centered columns
    to reduce g = n·(a_t grad_t + a_p grad_p). Returns a and centered D.
    """
    Dt = jnp.sum(Nn * grad_t_bdry, axis=1)
    Dp = jnp.sum(Nn * grad_p_bdry, axis=1)
    D = jnp.stack([Dt, Dp], axis=1)  # (N,2)

    # Weighted centering → columns zero-mean w.r.t. W
    Wsum = jnp.sum(W) + 1e-30
    mu = (W @ D) / Wsum               # (2,)
    D0 = D - mu[None, :]

    # Work with weighted data
    Wsqrt = jnp.sqrt(W)
    Dw0 = D0 * Wsqrt[:, None]

    # Leading singular vector: direction that reduces magnitude fastest
    U, S, Vt = jnp.linalg.svd(Dw0, full_matrices=False)
    a_dir = -Vt[0, :]

    # Scale so ||D0 a||_W2 hits a reasonable target (median column norm / 2)
    col_w2 = jnp.array([jnp.sqrt(jnp.dot(W, D0[:,0]**2)), jnp.sqrt(jnp.dot(W, D0[:,1]**2))])
    target = 0.5 * float(jnp.median(col_w2))
    denom = float(jnp.sqrt(jnp.dot(W, (D0 @ a_dir)**2)) + 1e-30)
    scale = target / denom
    a = scale * a_dir

    if verbose:
        g_fit = D0 @ a
        vec_stats("[MV-FIT] g(a)=n·∇φ_mv (centered)", g_fit, W)
        print(f"[MV-FIT] a_t={float(a[0]):.6g}, a_p={float(a[1]):.6g}")
    return a, D, D0
