"""Tie-breaker scan: score + w * z(x) with small w (reorders near-ties only), head-only (K) or full list.
For each feature, cross-fit the choice of w (including w=0) over the 5 user folds."""
import hk, numpy as np, sys

c = hk.Ctx(); r = c.col("rank"); s = c.score
K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
names = ["ucos", "hmax", "hmax_idx", "h1", "h5", "h10", "h32", "hmean5", "n_hdup95", "c_contact", "c_order", "c_fav", "c_lead",
         "c_vfeed", "c_vsearch", "cat_views", "cat_actions", "cat_newest_idx", "cat_user_rank", "cat_hist_share", "cand_maxcos",
         "n_cand_dup95", "cat_n_in_feed", "f1", "f2", "f3", "f4", "f5", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo",
         "it_hist_et3_luo", "it_hist_et4_luo", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "f2_home", "same_cat_newest",
         "s_ft", "s_to"]
inh = r <= K
ws = [-0.08, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08]
nvar = 0
out = []
for nm in names:
    x = c.col(nm).astype(float)
    if nm.startswith("it_hist") or nm in ("seq_cv_invd20", "n_hdup95", "n_cand_dup95", "cat_views", "cat_actions", "cat_n_in_feed"):
        x = np.log1p(np.maximum(x, 0))
    mu = np.nanmean(x[inh]); sd = np.nanstd(x[inh]) + 1e-12
    xz = np.nan_to_num((x - mu) / sd, nan=0.0)
    def mk(w):
        new = s.copy()
        new[inh] = 100 + s[inh] + w * xz[inh]
        return new + 1e-9 * s
    m, se, ch, oof, D = c.crossfit(mk, ws)
    nvar += len(ws)
    best_in = max(ws, key=lambda w: D[w].mean())
    out.append((nm, m, se, ch, best_in, D[best_in].mean()))
out.sort(key=lambda t: -t[1] / t[2])
print(f"K={K}, variants {nvar}")
for t in out:
    print(f"{t[0]:18s} cf={t[1]:+.5f} se={t[2]:.5f} z={t[1]/t[2]:+.2f}  chosen={t[3]}  in-sample best w={t[4]} {t[5]:+.5f}")
