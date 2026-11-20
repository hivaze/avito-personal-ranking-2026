"""Local matched comparison: each positive vs the negatives at ranks r-K..r+K in the same feed (score-matched).
Features that still differ are under-used by the model at that point of the list."""
import sys
from util import *

K = 3
df = load().sort(["fidx", "rank"])
NUM = ["f1", "f2", "f3", "f4", "f5", "xp_imp_c50", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo",
       "it_hist_et4_luo", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "ucos", "hmax", "hmax_idx", "h1", "h5", "h10", "h32",
       "hmean5", "n_hdup95", "c_contact", "c_order", "c_fav", "c_lead", "c_vfeed", "c_vsearch", "cat_views", "cat_actions",
       "cat_newest_idx", "cat_user_rank", "cat_hist_share", "cand_maxcos", "n_cand_dup95", "cat_n_in_feed", "cat_rank_in_feed",
       "s_ft", "s_to", "score", "same_cat_newest", "f2_home", "hmax_type", "hmax_src"]
X = {x: df[x].cast(pl.Float64).to_numpy() for x in NUM}
X["cold"] = (X["xp_imp_c50"] == 0).astype(float)
X["cur0"] = (X["cat_user_rank"] == 0).astype(float)
X["hty4"] = (X["hmax_type"] == 4).astype(float)
X["hty1"] = (X["hmax_type"] == 1).astype(float)
X["hsrc2"] = (X["hmax_src"] == 2).astype(float)
X["hidx0"] = (X["hmax_idx"] == 0).astype(float)
X["sdiff"] = X["s_ft"] - X["s_to"]
X["hd95pos"] = (X["n_hdup95"] > 0).astype(float)
X["cd95pos"] = (X["n_cand_dup95"] > 0).astype(float)
y = df["label"].to_numpy()
rank = df["rank"].to_numpy()
M = df["M"].to_numpy()
off, sz = kit.feed_layout(df)
fidr = np.repeat(np.arange(len(sz)), sz)
pos_idx = np.nonzero(y == 1)[0]
regions = {"r1-3": (1, 3), "r4-10": (4, 10), "r11-30": (11, 30), "r31-80": (31, 80), "r81+": (81, 999)}
for rname, (lo, hi) in regions.items():
    sel = pos_idx[(rank[pos_idx] >= lo) & (rank[pos_idx] <= hi)]
    diffs = {k: [] for k in X}
    for i in sel:
        o = off[fidr[i]]; m = sz[fidr[i]]
        j0 = max(o, i - K); j1 = min(o + m, i + K + 1)
        nb = np.arange(j0, j1)
        nb = nb[(y[nb] == 0)]
        if len(nb) == 0:
            continue
        for k, v in X.items():
            a = v[i]; b = v[nb]
            b = b[np.isfinite(b)]
            if np.isfinite(a) and len(b):
                diffs[k].append(a - b.mean())
    print(f"=== {rname}: positives {len(sel)}")
    out = []
    for k, d in diffs.items():
        d = np.array(d)
        if len(d) < 30:
            continue
        sd = np.nanstd(X[k][np.isfinite(X[k])])
        z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        out.append((abs(z), k, d.mean(), d.mean() / (sd + 1e-12), z, len(d)))
    for a, k, m, ms, z, nn in sorted(out, reverse=True)[:22]:
        print(f"   {k:18s} diff={m:+.4f} ({ms:+.3f} sd) z={z:+.2f} n={nn}")
