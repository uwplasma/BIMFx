import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.interpolate import griddata

from bimfx.MFS.geometry import orthonormal_complement

def pct(a, p): return float(np.percentile(np.asarray(a), p))

def fix_matplotlib_3d(ax):
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1]-y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1]-z_limits[0]); z_mid = np.mean(z_limits)
    R = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid-R, x_mid+R]); ax.set_ylim3d([y_mid-R, y_mid+R]); ax.set_zlim3d([z_mid-R, z_mid+R])

def plot_geometry_and_solution(P, N, grads_where, title_suffix="", show_normals=True, kind="torus", a_hat=None):
    """
    Left: surface + normals + scatter colored by |∇φ|
    Right: imshow of |∇φ| projected to best-fit (u,v) plane.
    """
    grad_mag = np.linalg.norm(np.asarray(grads_where), axis=1)

    fig = plt.figure(figsize=(14, 6))
    axL = fig.add_subplot(1, 2, 1, projection="3d")
    axR = fig.add_subplot(1, 2, 2)

    # --- Left: 3D with normals and colored points ---
    center = np.mean(np.asarray(P), axis=0)
    median_radius = np.median(np.linalg.norm(np.asarray(P) - center, axis=1))
    q_len = float(0.1 * median_radius)

    # Points colored by |∇φ|
    vmin = pct(grad_mag, 1); vmax = pct(grad_mag, 90)
    sc = axL.scatter(P[:,0], P[:,1], P[:,2],
                     c=grad_mag, s=6, cmap="viridis", vmin=vmin, vmax=vmax)
    # Normals
    if show_normals and N is not None:
        axL.quiver(P[:,0], P[:,1], P[:,2], N[:,0], N[:,1], N[:,2],
                   length=q_len, linewidth=0.5, normalize=True, color="k", alpha=0.7)
    axL.set_title(r"Surface & normals, colored by $|\nabla\phi|$" + title_suffix)
    axL.set_xlabel("x"); axL.set_ylabel("y"); axL.set_zlabel("z")
    axL.view_init(elev=20, azim=35); fix_matplotlib_3d(axL)
    cbar = fig.colorbar(sc, ax=axL, shrink=0.7, label=r"$|\nabla \phi|$ on $\Gamma_-$")

    # --- Right panel: coordinates for imshow ---
    # For torus: x = TRUE cylindrical φ := atan2(y,x); y = poloidal θ (tokamak-style)
    # For mirror: keep (s, θ) as before.
    if kind == "mirror":
        # keep axis-aware (s, θ)
        phi_a, theta, s = angles_for_axis(
            jnp.asarray(P), jnp.asarray(N), jnp.asarray(a_hat) if a_hat is not None else jnp.array([0.0,0.0,1.0]),
            center=jnp.asarray(np.mean(np.asarray(P), axis=0))
        )
        x_axis_vals = np.asarray(s) - np.median(np.asarray(s))
        x_label = "axial coordinate s"
        y_axis_vals = np.asarray(theta)
        y_label = "poloidal angle θ (rad)"
    else:
        # TORUS: true cylindrical φ on x-axis
        Pnp = np.asarray(P)
        x_axis_vals = np.arctan2(Pnp[:,1], Pnp[:,0])  # TRUE φ
        x_label = "toroidal angle ϕ (rad)"
        # Build θ the same robust way you already do (tokamak-style)
        phi, theta = _angles_phi_theta(P, N)
        y_axis_vals = np.asarray(theta)
        y_label = "poloidal angle θ (rad)"

    # Rasterize |∇φ| to a grid in (x_axis, θ)
    nX = 360 if kind != "mirror" else 240
    nY = 180
    Xu = np.linspace(-np.pi, np.pi, nX) if kind != "mirror" else np.linspace(x_axis_vals.min(), x_axis_vals.max(), nX)
    Yu = np.linspace(-np.pi, np.pi, nY)
    XX, YY = np.meshgrid(Xu, Yu, indexing="xy")

    pts = np.column_stack([x_axis_vals, y_axis_vals])
    grid = griddata(points=pts, values=grad_mag, xi=(XX, YY), method='linear')

    im = axR.imshow(grid, origin='lower',
                    extent=(Xu[0], Xu[-1], Yu[0], Yu[-1]),
                    aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axR.set_xlabel(x_label)
    axR.set_ylabel(y_label)
    axR.set_title(r"$|\nabla \phi|$ in axis-aware coords")
    fig.colorbar(im, ax=axR, shrink=0.8, label=r"$|\nabla \phi|$")

    plt.tight_layout(); plt.show()

def plot_boundary_condition_errors(P, N, grad_on_ring):
    n_dot_grad = jnp.sum(N * grad_on_ring, axis=1)
    grad_norm  = jnp.linalg.norm(grad_on_ring, axis=1)
    rn = jnp.abs(n_dot_grad) / jnp.maximum(1e-30, grad_norm)  # dimensionless
    ang_deg = jnp.degrees(jnp.arccos(jnp.clip(1.0 - 2.0*rn**2, -1.0, 1.0)))  # optional alt metric

    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(1,2,1, projection='3d')
    vmin = pct(rn, 1); vmax = pct(rn, 99)
    s1 = ax.scatter(P[:,0], P[:,1], P[:,2], c=np.array(rn), s=6, cmap='magma', vmin=vmin, vmax=vmax)
    fig.colorbar(s1, ax=ax, shrink=0.7, label=r"$|n\cdot\nabla\phi|/|\nabla\phi|$ on $\Gamma_-$")
    ax.set_title("Normalized Neumann residual on Γ₋")
    fix_matplotlib_3d(ax)

    ax2 = fig.add_subplot(1,2,2)
    idx = np.arange(P.shape[0])
    ax2.plot(idx, np.asarray(rn), lw=0.8, label=r"$|n\cdot\nabla\phi|/|\nabla\phi|$")
    ax2.set_xlabel("point index"); ax2.set_ylabel("dimensionless"); ax2.legend()
    ax2.set_title("BC residuals (line scan) on Γ₋")
    plt.tight_layout(); plt.show()


def plot_laplacian_errors_on_interior_band(P, lap_psi_at_interior, eps):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,2,1, projection='3d')
    vmin = pct(jnp.abs(lap_psi_at_interior), 1); vmax = pct(jnp.abs(lap_psi_at_interior), 99)
    s = ax.scatter(P[:,0], P[:,1], P[:,2],
                   c=np.array(jnp.abs(lap_psi_at_interior)), s=6, cmap='inferno',
                   vmin=vmin, vmax=vmax)
    fig.colorbar(s, ax=ax, shrink=0.7, label=rf"$|\nabla^2 \psi|(x - \varepsilon n),\ \varepsilon={eps:g}$")
    ax.set_title(r"Near-boundary Laplacian of $\psi$ (should be ≈0 inside)"); fix_matplotlib_3d(ax)
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(np.asarray(jnp.abs(lap_psi_at_interior)), bins=60, alpha=0.9)
    ax2.set_title(r"Histogram of $|\nabla^2 \psi|$ at interior offsets")
    ax2.set_xlabel("|∇²ψ|"); ax2.set_ylabel("count")
    plt.tight_layout(); plt.show()



# -------------------------- Diagnostics/plots -------------------- #
def _angles_phi_theta(P, N=None, verbose=False):
    P = np.asarray(P)
    x,y,z = P[:,0], P[:,1], P[:,2]
    phi = np.arctan2(y, x)

    if N is None:
        # Fallback: define θ by projecting ẑ to the local plane orthogonal to ϕ̂
        r2 = np.maximum(1e-30, x*x + y*y)
        phi_hat = np.stack([-y/np.sqrt(r2), x/np.sqrt(r2), np.zeros_like(x)], axis=1)
        zhat = np.array([0.0, 0.0, 1.0])[None,:]
        # Build an orthonormal frame {θ̂, ϕ̂, ϵ̂} with θ̂ ⟂ ϕ̂ and as close to ẑ as possible
        z_proj = zhat - (np.sum(zhat*phi_hat, axis=1, keepdims=True))*phi_hat
        z_proj /= (np.linalg.norm(z_proj, axis=1, keepdims=True) + 1e-30)
        theta_hat = z_proj
        # θ is angle of ẑ projection in (θ̂, ϕ̂) basis → 0 by construction
        theta = np.zeros_like(phi)
        return phi, theta

    # Existing path with true surface normals
    Nw = np.asarray(N)
    r2 = np.maximum(1e-30, x*x + y*y)
    phi_hat = np.stack([-y/np.sqrt(r2), x/np.sqrt(r2), np.zeros_like(x)], axis=1)
    phi_tan = phi_hat - np.sum(phi_hat*Nw, axis=1, keepdims=True)*Nw
    phi_tan /= (np.linalg.norm(phi_tan, axis=1, keepdims=True) + 1e-30)
    theta_hat = np.cross(Nw, phi_tan)
    theta_hat /= (np.linalg.norm(theta_hat, axis=1, keepdims=True) + 1e-30)
    zhat = np.array([0.0, 0.0, 1.0])[None,:]
    z_tan = zhat - np.sum(zhat*Nw, axis=1, keepdims=True)*Nw
    z_tan /= (np.linalg.norm(z_tan, axis=1, keepdims=True) + 1e-30)
    num = np.sum(z_tan * phi_tan, axis=1)
    den = np.sum(z_tan * theta_hat, axis=1)
    theta = np.arctan2(num, den)
    return phi, theta

def angles_for_axis(P, N, a_hat, center=None):
    """
    Azimuth ϕ_a and poloidal θ w.r.t. axis a_hat, using (e1,e2) ⟂ a_hat
    as a stable reference. Uses (X - center) everywhere.
    Returns (phi_a, theta, s).
    """
    Xw = jnp.asarray(P)
    Nw = jnp.asarray(N)
    if center is None:
        # fall back to geometric mean if not provided
        center = jnp.mean(Xw, axis=0)
    c = jnp.asarray(center)

    a = a_hat / jnp.maximum(1e-30, jnp.linalg.norm(a_hat))
    Xc = Xw - c[None, :]

    # orthonormal complement to a:
    e1_np, e2_np = orthonormal_complement(np.array(a_hat))
    e1 = jnp.asarray(e1_np); e2 = jnp.asarray(e2_np)

    # decompose r_perp in (e1,e2)
    r_par   = jnp.sum(Xc * a[None,:], axis=1, keepdims=True) * a[None,:]
    r_perp  = Xc - r_par
    u1 = jnp.sum(r_perp * e1[None,:], axis=1)
    u2 = jnp.sum(r_perp * e2[None,:], axis=1)
    phi_a = jnp.arctan2(u2, u1)  # azimuth around a

    # build φ̂_a and a tangent frame:
    r2 = jnp.maximum(1e-30, u1*u1 + u2*u2)[:, None]
    phi_hat = (u2[:,None]*e1[None,:] - u1[:,None]*e2[None,:]) / jnp.sqrt(r2)
    # project φ̂ to tangent plane and orthonormalize with N:
    phi_tan = phi_hat - jnp.sum(phi_hat * Nw, axis=1, keepdims=True) * Nw
    phi_tan = phi_tan / jnp.maximum(1e-30, jnp.linalg.norm(phi_tan, axis=1, keepdims=True))
    theta_hat = jnp.cross(Nw, phi_tan)
    theta_hat = theta_hat / jnp.maximum(1e-30, jnp.linalg.norm(theta_hat, axis=1, keepdims=True))

    # reference direction in tangent: project e1 to tangent
    e1_tan = e1[None,:] - jnp.sum(e1[None,:]*Nw, axis=1, keepdims=True)*Nw
    e1_tan = e1_tan / jnp.maximum(1e-30, jnp.linalg.norm(e1_tan, axis=1, keepdims=True))

    # θ is angle from e1_tan in the (phi_tan, theta_hat) basis
    num = jnp.sum(e1_tan * phi_tan, axis=1)
    den = jnp.sum(e1_tan * theta_hat, axis=1)
    theta = jnp.arctan2(num, den)

    # axial coordinate along a:
    s = jnp.sum(Xc * a[None,:], axis=1)
    return phi_a, theta, s