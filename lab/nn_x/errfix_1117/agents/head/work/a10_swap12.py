"""Rank-1 vs rank-2 swap scan: in which bins of the rank-1 item's (or the pair's) features is y2 > y1?
NDCG gain of a 1<->2 swap = (y2 - y1) * (1 - 1/log2(3)) / IDCG."""
import hk, numpy as np, polars as pl, sys

c = hk.Ctx(); df = c.df
A = int(sys.argv[1]) if len(sys.argv) > 1 else 1
B = int(sys.argv[2]) if len(sys.argv) > 2 else 2
r = c.col("rank")
i1 = np.nonzero(r == A)[0]; i2 = np.nonzero(r == B)[0]
assert (c.feed_of_row[i1] == c.feed_of_row[i2]).all()
npos = np.bincount(c.feed_of_row, weights=c.y)
idcg = np.array([hk._D[:int(k)].sum() for k in npos])
D = hk._D
g = (c.y[i2] - c.y[i1]) * (D[A - 1] - D[B - 1]) / idcg
print(f"swap {A}<->{B}: mean gain if always {g.mean():+.5f}; y{A}={c.y[i1].mean():.4f} y{B}={c.y[i2].mean():.4f}")
names = ["gap_next", "M", "H", "n_feed_cats", "n_hist_cats32", "ucos", "hmax", "hmax_idx", "h1", "h5", "hmean5", "n_hdup95", "c_contact",
         "c_fav", "c_vfeed", "c_vsearch", "cat_views", "cat_actions", "cat_newest_idx", "cat_hist_share", "cand_maxcos", "n_cand_dup95",
         "cat_n_in_feed", "f1", "f2", "f3", "f4", "f5", "it_hist_all_luo", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "s_ft", "s_to"]
out = []
for nm in names:
    x1 = c.col(nm).astype(float)[i1]; x2 = c.col(nm).astype(float)[i2]
    for lab, x in [(f"r{A}", x1), ("diff", x1 - x2), (f"ftto", None)]:
        if lab == "ftto":
            if nm != "s_ft":
                continue
            x = (c.col("s_ft") - c.col("s_to")).astype(float)
            x = x[i1] - x[i2]
            lab = "d(ft-to)"
        ok = np.isfinite(x)
        if ok.sum() < 500:
            continue
        qs = np.unique(np.quantile(x[ok], np.linspace(0, 1, 6)))
        b = np.digitize(x[ok], qs[1:-1])
        for k in np.unique(b):
            m = b == k
            if m.sum() < 300:
                continue
            gg = g[ok][m]
            out.append((nm, lab, f"[{x[ok][m].min():.3g},{x[ok][m].max():.3g}]", m.sum(), gg.mean(), gg.mean() / (gg.std() / np.sqrt(m.sum())),
                        c.y[i1][ok][m].mean(), c.y[i2][ok][m].mean()))
        if (~ok).sum() >= 300:
            gg = g[~ok]
            out.append((nm, lab, "NaN", (~ok).sum(), gg.mean(), gg.mean() / (gg.std() / np.sqrt((~ok).sum())), c.y[i1][~ok].mean(), c.y[i2][~ok].mean()))
out.sort(key=lambda t: -t[5])
print("bins tested", len(out))
for t in out[:25]:
    print(f"{t[0]:16s} {t[1]:8s} {t[2]:20s} n={t[3]:5d} gain/feed={t[4]:+.5f} z={t[5]:+.2f}  y{A}={t[6]:.3f} y{B}={t[7]:.3f}  total={t[4]*t[3]/len(i1):+.5f}")
