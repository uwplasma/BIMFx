import jax.numpy as jnp
from jax import lax, jit, tree_util
import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass

@tree_util.register_pytree_node_class
@dataclass
class ScaleInfo:
    center: jnp.ndarray   # (3,)
    scale: jnp.ndarray    # scalar array, shape ()

    def tree_flatten(self):
        # ensure scale is a JAX scalar
        return ((self.center, jnp.asarray(self.scale)), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        center, scale = children
        # IMPORTANT: do NOT call float(scale) here
        return cls(center=center, scale=jnp.asarray(scale))

def normalize_geometry(P, verbose=True):
    c = jnp.mean(P, axis=0)
    r = jnp.linalg.norm(P - c, axis=1)
    r_med = jnp.median(r)                   # keep as JAX scalar
    s = 1.0 / jnp.maximum(r_med, 1e-12)     # JAX scalar (shape ())
    Pn = (P - c) * s
    if verbose:
        print(f"[SCALE] center={np.array(c)}, median radius={float(np.asarray(r_med)):.6g}, "
              f"scale={float(np.asarray(s)):.6g} (so median radius→1)")
    return Pn, ScaleInfo(center=c, scale=s)

# -------------------- Geometry and angles ---------------------- #
def detect_geometry_and_axis(Pn, verbose=True):
    """
    Use PCA singular values to pick an axis and a geometry kind:
      - 'torus': s1 ~ s2 >> s3  → axis = e3  (smallest variance; surface normal to best-fit plane)
      - 'mirror': s1 >> s2 ~ s3 → axis = e1  (largest variance; long direction)
    If ambiguous, default to torus behavior (axis=e3).
    Returns (kind, a_hat [3,], E [3x3], singvals [3], center c).
    """
    c_np, E_np, s = best_fit_axis(np.array(Pn), verbose=False)
    s = np.asarray(s)  # s[0]>=s[1]>=s[2]
    E = np.asarray(E_np)                         # columns e1,e2,e3
    e1, e2, e3 = E[:,0], E[:,1], E[:,2]

    # Heuristics with margins (tune if needed)
    ratio_long = s[0] / max(s[1], 1e-12)
    ratio_thin = s[1] / max(s[2], 1e-12)

    if ratio_long > 2.0 and ratio_thin < 1.8:
        kind = "mirror"   # one long direction >> two comparable transverse
        a_hat = jnp.asarray(e1 / (np.linalg.norm(e1) + 1e-30))
    elif ratio_thin > 2.0 and ratio_long < 1.8:
        kind = "torus"    # thin shell: two wide directions >> one thin
        a_hat = jnp.asarray(e3 / (np.linalg.norm(e3) + 1e-30))
    else:
        # Roughly isotropic / ambiguous (e.g. sphere-like): treat as simply-connected.
        kind = "blob"
        a_hat = jnp.asarray(e1 / (np.linalg.norm(e1) + 1e-30))

    if verbose:
        print(f"[GEOM] s (desc) = {s}; ratio_long={ratio_long:.2f}, ratio_thin={ratio_thin:.2f}")
        print(f"[GEOM] kind={kind}, axis=a_hat = {np.array(a_hat)}")

    return kind, a_hat, jnp.asarray(E_np), jnp.asarray(c_np), jnp.asarray(s)

def best_fit_axis(points, verbose=True):
    c = np.mean(points, axis=0)
    X = points - c
    # SVD gives principal axes; singular values are already sorted desc.
    # X ≈ U diag(s) V^T, and rows of V^T are principal directions.
    _, svals, vt = np.linalg.svd(X, full_matrices=False)
    v1, v2, v3 = vt  # v1: largest variance axis, v3: smallest
    e3 = v3 / (np.linalg.norm(v3) + 1e-30)
    e1 = v1 / (np.linalg.norm(v1) + 1e-30)
    e2 = np.cross(e3, e1)
    E = np.stack([e1, e2, e3], axis=1)
    if verbose:
        print(f"[PCA] singular values (desc): {svals}")
        print("[AXES] Using e3 = smallest-singular-vector (best-fit plane normal).")
    return jnp.asarray(c), jnp.asarray(E), svals

@jit
def project_to_local(P, c, E): return (P - c) @ E

@jit
def cylindrical_angle_and_radius(local_pts):
    x, y = local_pts[:,0], local_pts[:,1]
    theta = jnp.arctan2(y, x)
    rho   = jnp.sqrt(jnp.maximum(1e-30, x*x + y*y))
    return theta, rho

def detect_if_angle_is_meaningful(theta, label):
    th = np.unwrap(np.array(theta)); span = np.max(th) - np.min(th)
    print(f"[MV] {label}: angular span ~ {span:.3f} rad (~{span*180/np.pi:.1f} deg)")
    return span > (200.0*np.pi/180.0)

def _rho_floor_from_points(Pn, c, E, frac=0.02):
    # project once and take a robust floor: a few percent of median radius
    Ploc = (Pn - c) @ E
    _, rho = cylindrical_angle_and_radius(Ploc[:, :2])
    rho_med = float(jnp.median(rho))
    return max(1e-3 * rho_med, frac * rho_med)  # safety + user-tunable

@jit
def grad_theta_world_from_plane(local_pts, E_plane_cols, rho_floor):
    e1 = E_plane_cols[:,0]; e2 = E_plane_cols[:,1]
    x, y = local_pts[:,0], local_pts[:,1]
    rho2 = x*x + y*y
    # soft clamp: rho_eff^2 = rho^2 + rho_floor^2  (keeps direction, limits magnitude)
    rho2_eff = rho2 + (rho_floor * rho_floor)
    a = -y / rho2_eff
    b =  x / rho2_eff
    return a[:,None]*e1[None,:] + b[:,None]*e2[None,:]

# -------------------------- kNN scales & weights ------------------------ #
def kNN_geometry_stats(Pn, k=48, verbose=True):
    c, E, _ = best_fit_axis(np.array(Pn), verbose=False)
    Ploc = project_to_local(Pn, c, E)
    XY = np.asarray(Ploc[:, :2])
    k_eff = min(k+1, len(XY))
    tree = cKDTree(XY)
    dists, _ = tree.query(XY, k=k_eff)
    rk = dists[:, -1]                              # k-th neighbor radius in local plane (includes self at 0)
    W = jnp.asarray(np.pi * rk**2, dtype=jnp.float64)
    if verbose:
        print(f"[QUAD] k-NN k={k}, area stats: min={float(W.min()):.3g}, max={float(W.max()):.3g}, median={float(jnp.median(W)):.3g}")
        print(f"[QUAD] k-NN radius stats: min={float(rk.min()):.3g}, max={float(rk.max()):.3g}, median={float(np.median(rk)):.3g}")
        print(f"[QUAD] total area estimate (sum W)≈{float(jnp.sum(W)):.3f}")
    return W, rk


def orthonormal_complement(a_hat):
    """
    Return two orthonormal vectors spanning the plane ⟂ a_hat.
    Deterministic construction.
    """
    a = np.asarray(a_hat) / (np.linalg.norm(a_hat) + 1e-30)
    # pick any vector not parallel to a
    t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = t - np.dot(t, a)*a
    e1 /= (np.linalg.norm(e1) + 1e-30)
    e2 = np.cross(a, e1)
    e2 /= (np.linalg.norm(e2) + 1e-30)
    return e1, e2


@jit
def grad_azimuth_about_axis(Xn, a_hat):
    """
    ∇ϕ_a for azimuth around an arbitrary unit axis a_hat.
      r_perp = X - (X·a)a
      ∇ϕ_a = (a × r_perp) / |r_perp|^2
    """
    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    r_par   = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
    r_perp  = Xn - r_par
    r2      = jnp.maximum(1e-30, jnp.sum(r_perp*r_perp, axis=1, keepdims=True))
    cross   = jnp.cross(a[None,:], r_perp)
    return cross / r2

def multivalued_bases_about_axis(Pn, Nn, a_hat, verbose=True):
    """
    Toroidal-like multivalued set defined around the chosen axis a_hat.
      - grad_t(X) = ∇ϕ_a(X) with azimuth around a_hat
      - grad_p(X) = θ̂(X): build from n × ϕ̂_tan, where ϕ̂_tan is azimuth unit projected to tangent plane
    This reduces to your old bases when a_hat = ẑ.
    """
    Pn_ref = jnp.asarray(Pn)
    Nn_ref = jnp.asarray(Nn)
    a_hat  = jnp.asarray(a_hat)

    @jit
    def _nearest_normal_jax(Xn):
        X2 = jnp.sum(Xn*Xn, axis=1, keepdims=True)
        P2 = jnp.sum(Pn_ref*Pn_ref, axis=1, keepdims=True)
        dist2 = X2 + P2.T - 2.0 * (Xn @ Pn_ref.T)
        idx = jnp.argmin(dist2, axis=1)
        return Nn_ref[idx, :]

    @jit
    def _unit(v, eps=1e-30):
        nrm = jnp.linalg.norm(v, axis=1, keepdims=True)
        return v / jnp.maximum(eps, nrm)

    @jit
    def _project_tangent(v, n):
        return v - jnp.sum(v*n, axis=1, keepdims=True)*n

    def grad_t(Xn):
        return grad_azimuth_about_axis(Xn, a_hat)

    def grad_p(Xn):
        # Build φ̂_a (azimuth unit) then project to tangent and make θ̂
        n = _nearest_normal_jax(Xn)
        # φ̂_a = unit(a × r_perp)
        a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
        r_par   = jnp.sum(Xn * a[None,:], axis=1, keepdims=True) * a[None,:]
        r_perp  = Xn - r_par
        phi_hat = _unit(jnp.cross(a[None,:], r_perp))
        phi_tan = _unit(_project_tangent(phi_hat, n))
        theta_hat = _unit(jnp.cross(n, phi_tan))
        return theta_hat

    def phi_t(Xn):
        # Unused value basis for consistency; keeping interface
        return jnp.zeros((Xn.shape[0],), dtype=jnp.float64)

    def phi_p(Xn):
        return jnp.zeros((Xn.shape[0],), dtype=jnp.float64)

    if verbose:
        print("[MV] Using axis-aware multivalued bases around detected axis a_hat.")

    return (phi_t, grad_t, phi_p, grad_p)

# -------------------------- Orientation check -------------------- #
def maybe_flip_normals(P, N):
    c = jnp.mean(P, axis=0)
    s = jnp.sum((P - c) * N, axis=1)
    avg = float(jnp.mean(s))
    if avg < 0:
        print(f"[ORIENT] Normals inward on average (⟨(P-c)·N⟩≈{avg:.3e}) → flipping.")
        return -N, True
    print(f"[ORIENT] Normals seem outward (⟨(P-c)·N⟩≈{avg:.3e}).")
    return N, False
