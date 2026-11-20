"""Head residuals: label minus the expected rate given (rank, M bucket), by feature bins, within ranks <= K.
Also rank-1 specifically: residual for rank-1 items by feature bins, and the rank-2 rate in the same feeds."""
import hk, numpy as np, polars as pl, sys

c = hk.Ctx(); df = c.df
K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
Mb = np.digitize(c.col("M"), [30, 60, 100, 200])
r = c.col("rank")
rb = np.minimum(r, 30)
key = Mb * 1000 + rb
# expected rate per (M bucket, rank)
uk, inv = np.unique(key, return_inverse=True)
rate = np.bincount(inv, weights=c.y) / np.bincount(inv)
exp = rate[inv]
res = c.y - exp
inh = r <= K
num = ["ucos", "hmax", "hmax_idx", "h1", "h5", "h10", "h32", "hmean5", "n_hdup95", "c_contact", "c_order", "c_fav", "c_lead",
       "c_vfeed", "c_vsearch", "cat_views", "cat_actions", "cat_newest_idx", "cat_user_rank", "cat_hist_share", "cand_maxcos",
       "n_cand_dup95", "cat_n_in_feed", "cat_rank_in_feed", "f1", "f2", "f3", "f4", "f5", "it_hist_all_luo", "it_hist_et1_luo",
       "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z",
       "gap_next", "gap_top", "H", "hour", "n_hist_cats32", "n_feed_cats"]
derived = {
    "d_ft_to": (c.col("s_ft") - c.col("s_to")).astype(float),
    "cat_newest_idx_pos": np.where(c.col("cat_newest_idx") < 0, 999, c.col("cat_newest_idx")).astype(float),
}
cat_like = ["cat", "hmax_type", "hmax_src", "newest_type", "newest_src", "guest", "f2_home", "same_cat_newest", "uvec_missing"]
out = []
n_tot = inh.sum()
for name in num + list(derived):
    x = derived[name] if name in derived else c.col(name).astype(float)
    xs = x[inh]; rs = res[inh]
    ok = np.isfinite(xs)
    nanmask = ~ok
    if nanmask.sum() > 200:
        out.append((name, "NaN", nanmask.sum(), rs[nanmask].mean(), rs[nanmask].mean() / (rs[nanmask].std() / np.sqrt(nanmask.sum()))))
    qs = np.unique(np.quantile(xs[ok], np.linspace(0, 1, 11)))
    b = np.digitize(xs[ok], qs[1:-1])
    for k in np.unique(b):
        m = b == k
        if m.sum() < 200:
            continue
        mu = rs[ok][m].mean(); se = rs[ok][m].std() / np.sqrt(m.sum())
        lo = xs[ok][m].min(); hi = xs[ok][m].max()
        out.append((name, f"[{lo:.3g},{hi:.3g}]", m.sum(), mu, mu / se))
for name in cat_like:
    x = c.col(name).astype(float)[inh]; rs = res[inh]
    for v in np.unique(x):
        m = x == v
        if m.sum() < 200:
            continue
        mu = rs[m].mean(); se = rs[m].std() / np.sqrt(m.sum())
        out.append((name, f"={v:g}", m.sum(), mu, mu / se))
out.sort(key=lambda t: -abs(t[4]))
print(f"head K={K}: rows {n_tot}, mean label {c.y[inh].mean():.4f}; bins tested {len(out)}")
for t in out[:45]:
    print(f"{t[0]:20s} {t[1]:22s} n={t[2]:6d} resid={t[3]:+.4f} z={t[4]:+.2f}")
