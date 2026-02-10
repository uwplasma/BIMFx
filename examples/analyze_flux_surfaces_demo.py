import numpy as np

from bimfx.fci import analyze_flux_surfaces


def _sphere_points(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    P = rng.normal(size=(n, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    N = P.copy()
    return P, N


def _B_helical(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.stack([-y, x, 0.2 * np.ones_like(z)], axis=1)


def main():
    xs = np.linspace(-1.0, 1.0, 48)
    ys = np.linspace(-1.0, 1.0, 48)
    zs = np.linspace(-1.0, 1.0, 48)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    psi = X**2 + Y**2 + Z**2

    P, N = _sphere_points(600)
    analysis = analyze_flux_surfaces(
        _B_helical,
        psi,
        xs,
        ys,
        zs,
        P=P,
        N=N,
        poincare_phi_planes=[0.0, 0.8],
        n_steps=1200,
        ds=0.02,
    )

    print(
        "psi_along shape:",
        analysis.psi_along.shape,
        "poincare points:",
        analysis.poincare.R.size,
    )

    try:
        from bimfx.fci import plot_poincare_overlays, plot_psi_along_fieldlines

        plot_poincare_overlays(analysis.poincare, analysis.psi_slices)
        plot_psi_along_fieldlines(analysis.s, analysis.psi_along)
    except ImportError as exc:
        print(exc)
        print("Install with: pip install 'bimfx[plot]'")


if __name__ == "__main__":
    main()
