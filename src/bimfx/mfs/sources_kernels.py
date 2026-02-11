from jax import jit, vmap, jacrev
import jax.numpy as jnp
import numpy as np

from bimfx.utils.fastsum import BarnesHut3D

# ----------------------------- Kernels -------------------------- #
@jit
def green_G(x, y):
    r = jnp.linalg.norm(x - y, axis=-1)
    return 1.0 / (4.0 * jnp.pi * jnp.maximum(1e-30, r))

@jit
def grad_green_x(x, y):
    r = x - y
    r2 = jnp.sum(r*r, axis=-1)
    r3 = jnp.maximum(1e-30, r2 * jnp.sqrt(r2))
    return - r / (4.0 * jnp.pi * r3[..., None])

# ------------------------- MFS source cloud ---------------------- #
def build_mfs_sources(Pn, Nn, rk, scale_info, source_factor=2.0, verbose=True):
    """
    Adaptive MFS sources: Y_i = P_i + δ_i N_i with δ_i = source_factor * rk_i
    where rk_i is the local kNN radius in the best-fit (u,v) plane.
    This makes δ smaller in the figure-8 neck and larger in the lobes.
    """
    # per-point δ_i
    delta_n_i = source_factor * rk.reshape(-1)         # (N,)
    Yn = Pn + delta_n_i[:, None] * Nn                  # (N,3)

    if verbose:
        dn_med = float(np.median(np.asarray(delta_n_i)))
        dn_min = float(np.min(np.asarray(delta_n_i)))
        dn_max = float(np.max(np.asarray(delta_n_i)))
        print(f"[MFS] Using adaptive source offsets δ_i (normalized): "
              f"median={dn_med:.4g}, min={dn_min:.4g}, max={dn_max:.4g}")

    return Yn, delta_n_i

# ----------------- Evaluators & Laplacian(ψ) ------------------- #
def build_evaluators_mfs(Pn, Yn, alpha, phi_t, phi_p, a, scinfo,
                         grad_t_fn, grad_p_fn):
    Y = Yn

    @jit
    def S_alpha_at(xn):
        Gvals = vmap(lambda y: green_G(xn, y))(Y)
        return jnp.dot(Gvals, alpha)

    @jit
    def grad_S_alpha_at(xn):
        Grads = vmap(lambda y: grad_green_x(xn, y))(Y)  # (M,3)
        return jnp.sum(Grads * alpha[:, None], axis=0)

    S_batch  = vmap(S_alpha_at)
    dS_batch = vmap(grad_S_alpha_at)

    @jit
    def phi_mv_world(X):
        Xn = (X - scinfo.center) * scinfo.scale
        return a[0]*phi_t(Xn) + a[1]*phi_p(Xn)

    # Multivalued gradients supplied by caller (already in WORLD basis)
    def grad_mv_world_batch(X):
        Xn = (X - scinfo.center) * scinfo.scale
        return scinfo.scale * (a[0]*grad_t_fn(Xn) + a[1]*grad_p_fn(Xn))

    @jit
    def psi_fn_world(X):
        Xn = (X - scinfo.center) * scinfo.scale
        return S_batch(Xn)

    @jit
    def grad_psi_fn_world(X):
        Xn = (X - scinfo.center) * scinfo.scale
        return (scinfo.scale) * dS_batch(Xn)

    @jit
    def phi_fn_world(X):
        return phi_mv_world(X) + psi_fn_world(X)

    def grad_fn_world(X):
        return grad_mv_world_batch(X) + grad_psi_fn_world(X)

    def laplacian_psi_world(X):
        def grad_at_point(x):
            return grad_psi_fn_world(x[None, :])[0]
        J = vmap(jacrev(grad_at_point))(X)   # (M,3,3)
        return jnp.trace(J, axis1=1, axis2=2)

    return phi_fn_world, grad_fn_world, psi_fn_world, grad_psi_fn_world, laplacian_psi_world, grad_mv_world_batch


def build_evaluators_mfs_accel(
    Pn,
    Yn,
    alpha,
    phi_t,
    phi_p,
    a,
    scinfo,
    grad_t_fn,
    grad_p_fn,
    *,
    theta: float = 0.6,
    leaf_size: int = 64,
):
    """Accelerated MFS evaluators using a Barnes-Hut tree for source sums."""
    tree = BarnesHut3D(np.asarray(Yn), np.asarray(alpha), theta=theta, leaf_size=leaf_size)

    def _phi_mv_world(X):
        Xn = (np.asarray(X) - np.asarray(scinfo.center)) * float(scinfo.scale)
        return np.asarray(a[0] * phi_t(jnp.asarray(Xn)) + a[1] * phi_p(jnp.asarray(Xn)))

    def _grad_mv_world_batch(X):
        Xn = (np.asarray(X) - np.asarray(scinfo.center)) * float(scinfo.scale)
        grad = scinfo.scale * (a[0] * grad_t_fn(jnp.asarray(Xn)) + a[1] * grad_p_fn(jnp.asarray(Xn)))
        return np.asarray(grad)

    def psi_fn_world(X):
        Xn = (np.asarray(X) - np.asarray(scinfo.center)) * float(scinfo.scale)
        return tree.potential(Xn)

    def grad_psi_fn_world(X):
        Xn = (np.asarray(X) - np.asarray(scinfo.center)) * float(scinfo.scale)
        grad_n = tree.gradient(Xn)
        return float(scinfo.scale) * grad_n

    def phi_fn_world(X):
        return _phi_mv_world(X) + psi_fn_world(X)

    def grad_fn_world(X):
        return _grad_mv_world_batch(X) + grad_psi_fn_world(X)

    def laplacian_psi_world(X):
        Xn = (np.asarray(X) - np.asarray(scinfo.center)) * float(scinfo.scale)
        eps = 1e-6
        lap = np.zeros(Xn.shape[0])
        for i, x in enumerate(Xn):
            phi0 = tree.potential(x)
            acc = 0.0
            for d in range(3):
                xp = x.copy()
                xm = x.copy()
                xp[d] += eps
                xm[d] -= eps
                acc += (tree.potential(xp) - 2 * phi0 + tree.potential(xm)) / (eps**2)
            lap[i] = acc * (scinfo.scale**2)
        return lap

    return phi_fn_world, grad_fn_world, psi_fn_world, grad_psi_fn_world, laplacian_psi_world, _grad_mv_world_batch
