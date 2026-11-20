"""Per-feed Pearson of final scores on TEST (and val) between pools (context for pick-2 diversity)."""
import sys; sys.path.insert(0, "<repo>/src")
import numpy as np, numba as nb
from common import feed_meta
from stack_val import feed_rank_z
P = "<repo>/preds"
@nb.njit(parallel=True)
def pf_corr(a, b, off, M):
    out = np.zeros(len(M))
    for f in nb.prange(len(M)):
        x = a[off[f]:off[f] + M[f]]; y = b[off[f]:off[f] + M[f]]
        x = x - x.mean(); y = y - y.mean(); out[f] = (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12)
    return out
for split in ["val", "test"]:
    _, M, o = feed_meta(split); M = M.astype(np.int64); o = o.astype(np.int64)
    z = lambda n: feed_rank_z(o, M, np.load(f"{P}/{n}.npy"))[:, 1].astype(np.float64)
    S = {t: 0.5 * z(f"ens_{t}_ft_{split}") + 0.5 * z(f"ens_{t}_to_{split}") for t in ["PF7x", "G3", "M2", "hfitP11W1to"]}
    for a, b in [("hfitP11W1to", "G3"), ("hfitP11W1to", "PF7x"), ("G3", "PF7x"), ("M2", "G3")]:
        print(split, a, b, f"{pf_corr(S[a], S[b], o, M).mean():.4f}")
