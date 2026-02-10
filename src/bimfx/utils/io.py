import numpy as np
import jax.numpy as jnp

def get_candidates(script_dir, file_name, subdir="inputs"):
    try:
        candidate_xyz = (script_dir / ".." / subdir / (file_name + ".csv")).resolve()
        candidate_normals = (script_dir / ".." / subdir / (file_name+'_normals.csv')).resolve()
        if candidate_xyz.exists():
            candidate_xyz = str(candidate_xyz)
            print(f"Resolved checkpoint path -> {candidate_xyz}")
        else: print(f"[WARN] Expected checkpoint not found at {candidate_xyz}; using provided path: {candidate_xyz}")
        if candidate_normals.exists():
            candidate_normals = str(candidate_normals)
            print(f"Resolved checkpoint path -> {candidate_normals}")
        else: print(f"[WARN] Expected checkpoint not found at {candidate_normals}; using provided path: {candidate_normals}")
    except Exception as e:
        print(f"[WARN] Failed to resolve ../{subdir} path: {e}; using provided path: {candidate_xyz}")
    return candidate_xyz, candidate_normals

# ----------------------------- I/O ----------------------------- #
def load_surface_xyz_normals(xyz_csv, nrm_csv, verbose=True):
    P = np.loadtxt(xyz_csv, delimiter=",", skiprows=1)
    N = np.loadtxt(nrm_csv, delimiter=",", skiprows=1)
    assert P.shape[1] == 3 and N.shape[1] == 3, "CSV must have 3 columns"
    nrm = N / np.maximum(1e-15, np.linalg.norm(N, axis=1, keepdims=True))
    if verbose:
        print(f"[LOAD] points: {P.shape}, normals: {N.shape}")
        print(f"[LOAD] point extents (min..max) per axis:")
        for k, nm in enumerate("xyz"):
            print(f"       {nm}: {P[:,k].min():.6g} .. {P[:,k].max():.6g}")
        nlen = np.linalg.norm(nrm, axis=1)
        print(f"[LOAD] normal lengths: min={nlen.min():.3g}, max={nlen.max():.3g}, mean={nlen.mean():.3g}")
    return jnp.asarray(P, dtype=jnp.float64), jnp.asarray(nrm, dtype=jnp.float64)
