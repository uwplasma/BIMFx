import numpy as np

from bimfx import solve_mfs
from bimfx.validation import validate_vacuum_field


def _torus_point_cloud(R: float, r: float, nphi: int, ntheta: int) -> tuple[np.ndarray, np.ndarray]:
    phis = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    thetas = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    points = []
    normals = []
    for phi in phis:
        cphi, sphi = np.cos(phi), np.sin(phi)
        for th in thetas:
            cth, sth = np.cos(th), np.sin(th)
            x = (R + r * cth) * cphi
            y = (R + r * cth) * sphi
            z = r * sth
            points.append([x, y, z])
            normals.append([cth * cphi, cth * sphi, sth])
    P = np.asarray(points, dtype=float)
    N = np.asarray(normals, dtype=float)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return P, N


def main():
    P, N = _torus_point_cloud(R=3.0, r=1.0, nphi=16, ntheta=16)
    field = solve_mfs(P, N, toroidal_flux=1.0)

    xs = np.linspace(-4.5, 4.5, 48)
    ys = np.linspace(-4.5, 4.5, 48)
    zs = np.linspace(-2.0, 2.0, 32)

    stats = validate_vacuum_field(field.B, P, N, xs, ys, zs)
    print(stats)


if __name__ == "__main__":
    main()
