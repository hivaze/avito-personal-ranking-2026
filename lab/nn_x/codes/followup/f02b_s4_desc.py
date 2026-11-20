"""Information only (not a gate): how much the s4 graph could matter at all. For val candidates with s4_link_any = 1 / s4_sym > 0:
row count, positives, target rate vs the same feeds' base rate, G3 act/exp (within-feed decile expectation, user-clustered SE)."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/followup")
from fu_common import *
V = Split("val"); y, fid, n = V.y, V.fid, V.n
g = np.load(P / "ens_G3_final_val.npy").astype(np.float64)
o = np.lexsort((g, fid)); fs_ = np.r_[True, fid[o][1:] != fid[o][:-1]]
rk = np.arange(len(o)) - np.flatnonzero(fs_)[np.cumsum(fs_) - 1]
pct = np.empty(len(o)); pct[o] = rk / np.maximum(V.M[fid[o]] - 1, 1)
dec = np.clip((pct * 10).astype(int), 0, 9)
rate = np.bincount(dec, y, minlength=10) / np.bincount(dec, minlength=10); ex = rate[dec]
ex = ex * (np.bincount(fid, y, minlength=n) / np.maximum(np.bincount(fid, ex, minlength=n), 1e-12))[fid]
for k in ["s4_link_any", "s4_sym", "s4_link"]:
    x = np.load(OUT / f"val_{k}.npy") > 0
    fe = np.bincount(fid, x, n) > 0
    base_rate = y[fe[fid] & ~x].mean()
    num = np.bincount(fid, y * x, n); den = np.bincount(fid, ex * x, n)
    r, se = ratio_cl(num, den, V.cl)
    # G3 within-feed percentile of linked positives
    pp = x & (y > 0)
    print(f"{k:12s}: rows {x.sum():,} ({x.mean():.4%}) in {fe.sum():,} feeds; positives {int(y[x].sum())} ({y[x].sum()/y.sum():.3%} of all val positives); "
          f"target rate {y[x].mean():.4f} vs other rows of the same feeds {base_rate:.4f}; G3 act/exp {r:.3f} +- {se:.3f}; "
          f"G3 within-feed percentile of linked positives: median {np.median(pct[pp]) if pp.any() else float('nan'):.3f}")
