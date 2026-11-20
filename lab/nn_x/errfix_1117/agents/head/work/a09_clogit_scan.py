"""Within-feed conditional-logit scan in the head: which feature carries signal beyond the score?"""
import hk, clogit, numpy as np, polars as pl, sys

c = hk.Ctx(); df = c.df
K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
r = c.col("rank"); inh = r <= K
fid = c.col("fidx")
P, S, Soff = clogit.build_sets(fid, c.y, inh)
print(f"K={K}: choice sets {len(P)}, mean set size {np.diff(Soff).mean():.2f}")
sc = c.score
b, se, ll0 = clogit.fit(sc[:, None], P, S, Soff)
print("score only beta", b, se, "ll", ll0)
names = ["s_ft", "s_to", "ucos", "hmax", "hmax_idx", "h1", "h5", "h10", "h32", "hmean5", "n_hdup95", "c_contact", "c_order", "c_fav",
         "c_lead", "c_vfeed", "c_vsearch", "cat_views", "cat_actions", "cat_newest_idx", "cat_user_rank", "cat_hist_share",
         "cand_maxcos", "n_cand_dup95", "cat_n_in_feed", "cat_rank_in_feed", "f1", "f2", "f3", "f4", "f5", "it_hist_all_luo",
         "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo", "xp_imp_c50", "seq_cv_invd20",
         "pre2_fe_z", "pre2_new_z", "f2_home", "same_cat_newest"]
res = []
for nm in names:
    x = c.col(nm).astype(float)
    if nm.startswith("it_hist") or nm in ("seq_cv_invd20", "n_hdup95", "n_cand_dup95", "cat_views", "cat_actions", "cat_n_in_feed"):
        x = np.log1p(np.maximum(x, 0))
    isn = ~np.isfinite(x)
    mu = np.nanmean(x[inh]); sd = np.nanstd(x[inh])
    xz = np.where(isn, 0.0, (x - mu) / (sd + 1e-12))
    cols = [sc, xz]
    if isn[inh].mean() > 0.01:
        cols.append(isn.astype(float))
    X = np.column_stack(cols)
    b, se_, ll = clogit.fit(X, P, S, Soff)
    res.append((nm, b[1], b[1] / se_[1], 2 * (ll - ll0), b[2] / se_[2] if len(b) > 2 else np.nan))
res.sort(key=lambda t: -abs(t[2]))
for t in res:
    print(f"{t[0]:18s} beta={t[1]:+.3f} z={t[2]:+.2f} LR={t[3]:.1f}  nan_z={t[4]:+.2f}")
