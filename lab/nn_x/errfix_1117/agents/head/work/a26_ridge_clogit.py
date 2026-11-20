"""Upper bound for LINEAR head corrections: ridge conditional logit with score + all features, fit on 4 folds,
re-rank the head of the 5th fold by the fitted utility (head set fixed)."""
import hk, clogit, numpy as np, sys

c = hk.Ctx(); r = c.col("rank"); fid = c.col("fidx"); sc = c.score
K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
names = ["s_ft", "ucos", "hmax", "hmax_idx", "h1", "h5", "h10", "h32", "hmean5", "n_hdup95", "c_contact", "c_order", "c_fav", "c_lead",
         "c_vfeed", "c_vsearch", "cat_views", "cat_actions", "cat_newest_idx", "cat_user_rank", "cat_hist_share", "cand_maxcos",
         "n_cand_dup95", "cat_n_in_feed", "f1", "f2", "f3", "f4", "f5", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo",
         "it_hist_et3_luo", "it_hist_et4_luo", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "f2_home", "same_cat_newest"]
inh = r <= K
cols = [sc]
for nm in names:
    x = c.col(nm).astype(float)
    if nm == "s_ft":
        x = (c.col("s_ft") - c.col("s_to")).astype(float)
    if nm.startswith("it_hist") or nm in ("seq_cv_invd20", "n_hdup95", "n_cand_dup95", "cat_views", "cat_actions", "cat_n_in_feed"):
        x = np.log1p(np.maximum(x, 0))
    isn = ~np.isfinite(x)
    mu = np.nanmean(x[inh]); sd = np.nanstd(x[inh]) + 1e-12
    cols.append(np.where(isn, 0, (x - mu) / sd))
    if isn[inh].mean() > 0.02:
        cols.append(isn.astype(float))
X = np.column_stack(cols)
print("K", K, "design", X.shape)
for l2 in [1e-2, 1e-1, 3e-1, 1.0]:
    new = sc.copy()
    for f in range(5):
        tr = inh & (c.fold_row != f); te = inh & (c.fold_row == f)
        P, S, Soff = clogit.build_sets(fid, c.y, tr)
        b, se, ll = clogit.fit(X, P, S, Soff, l2=l2, free0=True)
        new[te] = 100 + X[te] @ b
    new[inh] += 1e-9 * sc[inh]
    m, s_, d = c.delta(new)
    print(f"  l2={l2}: OOF delta {m:+.5f} se {s_:.5f} z {m/s_:+.2f}  folds", [round(d[c.fold_feed == f].mean() * 1e4, 1) for f in range(5)])
