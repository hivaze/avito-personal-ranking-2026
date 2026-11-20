"""ADDENDUM 1 (PREDECLARED.txt, 05:57:30 UTC; post-hoc, information only): x = f5 * 1[f4 >= 0.6801] through the x02 harness on val,
its tail win rate, and G3 act/exp by f5 quintile inside f4-T3 rows."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/followup")
from fu_common import *
V = Split("val"); y = V.y
g = np.load(P / "ens_G3_final_val.npy").astype(np.float64); gz = V.fz(g)
f4 = np.load(FLAT / "val/cand_f4.npy").astype(np.float64); f5 = np.load(FLAT / "val/cand_f5.npy").astype(np.float64)
t3 = f4 >= 0.6801
x = f5 * t3
dl, ws = nested(V, gz, x); se = dl.std() / np.sqrt(V.n)
tw = Tail(V, g).win(x)
print(f"f5_x_oldT3: rows in T3 {t3.mean():.4f}; val nested {dl.mean():+.5f} ({se:.5f}, z {dl.mean()/se:+.2f}) folds {ws}; "
      f"tail win {tw['win']:.4f} +- {tw['se']:.4f} (untied share {tw['untied_share']:.3f}, win|untied {tw['win_untied']:.4f})")
# G3 act/exp by f5 quintile inside T3 rows (expected from G3 within-feed deciles over all rows)
fid, n, M = V.fid, V.n, V.M
o = np.lexsort((gz, fid)); fs_ = np.r_[True, fid[o][1:] != fid[o][:-1]]
rk = np.arange(len(o)) - np.flatnonzero(fs_)[np.cumsum(fs_) - 1]
pct = np.empty(len(o)); pct[o] = rk / np.maximum(M[fid[o]] - 1, 1)
dec = np.clip((pct * 10).astype(int), 0, 9)
rate = np.bincount(dec, y, minlength=10) / np.bincount(dec, minlength=10); ex = rate[dec]
ex = ex * (np.bincount(fid, y, minlength=n) / np.maximum(np.bincount(fid, ex, minlength=n), 1e-12))[fid]
q = np.quantile(f5[t3], [0.2, 0.4, 0.6, 0.8]); b = np.searchsorted(np.unique(q), f5, side="right")
out = []
for i in range(b[t3].max() + 1):
    m = t3 & (b == i)
    if m.sum() == 0: continue
    r, s = ratio_cl(np.bincount(fid, y * m, n), np.bincount(fid, ex * m, n), V.cl)
    out.append(f"q{i} {m.sum()/t3.sum():.2f}: {r:.3f}+-{s:.3f}")
print("G3 act/exp by f5 quintile within f4-T3 rows (share: act/exp +- SE): " + " | ".join(out))
