"""Reviewer: row 99 from the stored nested predictions, own code."""
import numpy as np, polars as pl
R = "<repo>"
f = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); y = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)
def nd(s):
    n = len(M); fid = np.repeat(np.arange(n), M); off = np.concatenate([[0], np.cumsum(M)[:-1]])
    o = np.lexsort((y, -s.astype(np.float64), fid)); rk = np.arange(len(s)) - np.repeat(off, M); d = 1.0 / np.log2(rk + 2.0)
    dcg = np.bincount(fid, weights=y[o] * d, minlength=n); npos = np.bincount(fid, weights=y, minlength=n)
    return dcg / np.bincount(fid, weights=(rk < np.repeat(npos, M)) * d, minlength=n)
def z(s):
    fid = np.repeat(np.arange(len(M)), M); mu = np.bincount(fid, weights=s) / M; sd = np.sqrt(np.maximum(np.bincount(fid, weights=(s - mu[fid]) ** 2) / M, 1e-12)); return (s - mu[fid]) / sd[fid]
pse = lambda d: f"{d.mean():+.5f} +- {d.std(ddof=1) / np.sqrt(len(d)):.5f} (z {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f})"
D, Zi, Zii, Zp, Dp = [], 0, 0, 0, []
for v in ("v15_cd_all_e10", "v14_s93", "v17_pre2s_s102"):
    q = np.load(f"{R}/nn_x/solvers/preds/refit/nested_{v}.npz"); pi, pii = q["P_i"].astype(np.float64), q["P_ii"].astype(np.float64)
    pp = np.load(f"{R}/preds/nn_{v}_n2cpv3_val_oof.npy").astype(np.float64)
    a, b, c = nd(pi), nd(pii), nd(pp); D.append(b - a); Dp.append(c - a); Zi = Zi + z(pi); Zii = Zii + z(pii); Zp = Zp + z(pp)
    print(f"{v}: (i) {a.mean():.5f} (ii) {b.mean():.5f} prod {c.mean():.5f}; (ii)-(i) {pse(b - a)}; outer fold sizes {np.bincount(q['outer']).tolist()}")
print("row 99, pooled per-feed mean of the 3 member deltas (ii)-(i):", pse(np.mean(D, 0)), "n", len(M))
print("production OOF (another 80% seed) - (i):", pse(np.mean(Dp, 0)))
a, b, c = nd(Zi / 3), nd(Zii / 3), nd(Zp / 3)
print(f"pool level (z-mean of 3 members): (i) {a.mean():.5f} (ii) {b.mean():.5f} prod {c.mean():.5f}; D_pool {pse(b - a)}; prod - (i) {pse(c - a)}; (ii+prod)/2 - (i) {pse(nd((Zii + Zp) / 6) - a)}")
