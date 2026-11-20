"""Conditional-logit interaction scan in the head: score + x + x*1[segment] for feed segments.
Reports interaction z; with ~300 tests the null max |z| is ~3.3."""
import hk, clogit, numpy as np, polars as pl, sys

c = hk.Ctx(); r = c.col("rank"); fid = c.col("fidx"); sc = c.score
K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
inh = r <= K
P, S, Soff = clogit.build_sets(fid, c.y, inh)
nt = c.col("newest_type")
segs = {
    "nt_view": nt == 1, "nt_contact": np.isin(nt, [2, 3]), "nt_fav": nt == 4, "nt_order": np.isin(nt, [5, 6, 8]),
    "guest": c.col("guest").astype(bool), "M<=30": c.col("M") <= 30, "M>150": c.col("M") > 150,
    "H<=50": c.col("H") <= 50, "nhc32<=2": c.col("n_hist_cats32") <= 2, "night": (c.col("hour") <= 6),
    "src_search": c.col("newest_src") == 2, "src_feed": c.col("newest_src") == 1,
}
names = ["ucos", "hmax", "hmax_idx", "h1", "h5", "hmean5", "n_hdup95", "c_contact", "c_fav", "c_vfeed", "c_vsearch", "cat_views",
         "cat_actions", "cat_newest_idx", "cat_hist_share", "cand_maxcos", "n_cand_dup95", "cat_n_in_feed", "f1", "f2", "f3", "f4", "f5",
         "it_hist_all_luo", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "same_cat_newest", "d_ft_to"]
out = []
for nm in names:
    if nm == "d_ft_to":
        x = (c.col("s_ft") - c.col("s_to")).astype(float)
    else:
        x = c.col(nm).astype(float)
    if nm.startswith("it_hist") or nm in ("seq_cv_invd20", "n_hdup95", "n_cand_dup95", "cat_views", "cat_actions", "cat_n_in_feed"):
        x = np.log1p(np.maximum(x, 0))
    mu = np.nanmean(x[inh]); sd = np.nanstd(x[inh]) + 1e-12
    xz = np.nan_to_num((x - mu) / sd, nan=0.0)
    for sn, sm in segs.items():
        if sm[P].sum() < 150:
            continue
        X = np.column_stack([sc, xz, xz * sm])
        b, se, ll = clogit.fit(X, P, S, Soff)
        out.append((nm, sn, int(sm[P].sum()), b[1], b[2], b[2] / se[2]))
out.sort(key=lambda t: -abs(t[5]))
print(f"K={K} tests={len(out)}")
for t in out[:25]:
    print(f"{t[0]:16s} x {t[1]:11s} npos_seg={t[2]:5d} main={t[3]:+.3f} inter={t[4]:+.3f} z={t[5]:+.2f}")
