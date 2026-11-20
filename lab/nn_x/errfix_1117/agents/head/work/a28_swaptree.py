"""Cross-fitted swap discovery: regression tree on the gain of swapping ranks A and B (features of both items,
their difference, feed features). Swap only in leaves whose training-fold mean gain is > 0. Evaluate OOF."""
import hk, numpy as np, polars as pl, sys
from sklearn.tree import DecisionTreeRegressor

c = hk.Ctx(); df = c.df
A, B = int(sys.argv[1]), int(sys.argv[2])
r = c.col("rank")
i1 = np.nonzero(r == A)[0]; i2 = np.nonzero(r == B)[0]
npos = np.bincount(c.feed_of_row, weights=c.y)
idcg = np.array([hk._D[:int(k)].sum() for k in npos])
g = (c.y[i2] - c.y[i1]) * (hk._D[A - 1] - hk._D[B - 1]) / idcg
names = ["ucos", "hmax", "hmax_idx", "h1", "h5", "hmean5", "n_hdup95", "c_contact", "c_fav", "c_vfeed", "c_vsearch", "cat_views",
         "cat_actions", "cat_newest_idx", "cat_hist_share", "cand_maxcos", "n_cand_dup95", "cat_n_in_feed", "f1", "f2", "f3", "f4", "f5",
         "it_hist_all_luo", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "s_ft", "s_to", "gap_next"]
feed = ["M", "H", "n_hist_cats32", "n_feed_cats", "newest_type", "guest"]
F = []
for nm in names:
    x = np.nan_to_num(c.col(nm).astype(float), nan=-2.0)
    F += [x[i1], x[i2], x[i1] - x[i2]]
for nm in feed:
    F.append(c.col(nm).astype(float)[i1])
F.append((c.col("cat")[i1] == c.col("cat")[i2]).astype(float))
F = np.column_stack(F)
ff = c.fold_feed
for leaf in [200, 400, 800]:
    for depth in [2, 3, 4]:
        oof = np.zeros(len(g))
        nsw = 0
        for f in range(5):
            tr = ff != f; te = ff == f
            t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(F[tr], g[tr])
            lv_tr = t.apply(F[tr]); lv_te = t.apply(F[te])
            good = [l for l in np.unique(lv_tr) if g[tr][lv_tr == l].mean() > 0]
            sw = np.isin(lv_te, good)
            nsw += sw.sum()
            oof[te] = np.where(sw, g[te], 0.0)
        se = oof.std(ddof=1) / np.sqrt(len(oof))
        print(f"swap {A}<->{B} leaf={leaf} depth={depth}: OOF {oof.mean():+.5f} se {se:.5f} z {oof.mean()/se if se>0 else 0:+.2f} swapped feeds {nsw}")
