import numpy as np

def vec_stats(title, v, W=None):
    v_np = np.asarray(v)
    if W is None:
        print(f"[STATS] {title}: L2={np.linalg.norm(v_np):.3e}, "
              f"Linf={np.max(np.abs(v_np)):.3e}, mean={np.mean(v_np):.3e}")
    else:
        W_np = np.asarray(W)
        lw2 = np.sqrt(np.dot(W_np, v_np**2))
        mean_w = np.dot(W_np, v_np) / (np.sum(W_np) + 1e-30)
        print(f"[STATS] {title}: ||·||_W2={lw2:.3e}, Linf={np.max(np.abs(v_np)):.3e}, <W,·>={mean_w:.3e}")
