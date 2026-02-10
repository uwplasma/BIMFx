import numpy as np

from bimfx.fci import fit_flux_surfaces


def main():
    xs = np.linspace(-1.0, 1.0, 48)
    ys = np.linspace(-1.0, 1.0, 48)
    zs = np.linspace(-1.0, 1.0, 48)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    psi = X**2 + Y**2 + Z**2

    try:
        surfaces = fit_flux_surfaces(psi, xs, ys, zs, levels=[0.25, 0.5, 0.75])
    except ImportError as exc:
        print(exc)
        print("Install with: pip install 'bimfx[fci]'")
        return

    for surf in surfaces:
        print(
            f"level={surf.level:.2f} "
            f"verts={surf.vertices.shape[0]} "
            f"faces={surf.faces.shape[0]} "
            f"centroid={surf.centroid}"
        )


if __name__ == "__main__":
    main()
