"""Redundant dense inputs: pairs of the 104 inputs with |Pearson| >= 0.98 on val (row level) and on the within-feed
demeaned values (what matters for the metric).  CPU only."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "12"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s02_stats import NAMES, OUT
import numpy as np
X = np.load(OUT / "val_X.npy").astype(np.float64); mt = np.load(OUT / "val_meta.npz"); M, off = mt["M"], mt["off"]
fid = np.repeat(np.arange(len(M)), M)
mu = np.zeros((len(M), X.shape[1]))
np.add.at(mu, fid, X); mu /= M[:, None]
Xw = X - mu[fid]
for nm, A in [("row", X), ("within-feed", Xw)]:
    sd = A.std(0); ok = sd > 1e-9
    C = np.corrcoef(A[:, ok].T); idx = np.where(ok)[0]
    pairs = [(NAMES[idx[i]], NAMES[idx[j]], C[i, j]) for i in range(len(idx)) for j in range(i + 1, len(idx)) if abs(C[i, j]) >= 0.98]
    print(f"== {nm}: {len(pairs)} pairs with |r| >= 0.98")
    for a, b, r in pairs:
        print(f"   {a:24s} {b:24s} {r:+.4f}")
