"""Print individual feeds compactly: head + positives with lens and context columns."""
import sys
from lib import *
D = Data(); df = D.df
rank = D.col("rank")
best = np.minimum.reduceat(np.where(D.y == 1, rank, 1e9), D.off)
kind = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(seed)
G = df["guest"].to_numpy()[D.off]; H = df["H"].to_numpy()[D.off]
pools = {"bad": best >= 30, "good": best <= 3, "guest": G, "shortH": (H < 30) & ~G, "all": best > 0}
pool = np.nonzero(pools[kind])[0]
pick = rng.choice(pool, min(n, len(pool)), replace=False)
show = [("rank", "%4d"), ("label", "%1d"), ("score", "%5.2f"), ("s_ft", "%5.2f"), ("s_to", "%5.2f"), ("cat", "%3d"), ("cat_user_rank", "%3d"),
        ("cat_newest_idx", "%4d"), ("xp_imp_c50", "%4.1f"), ("it_hist_all_luo", "%5.0f"), ("f4", "%4.2f"), ("f5", "%4.2f"), ("f1", "%4.2f"),
        ("f2_home", "%1d"), ("seq_cv_invd20", "%5.2f"), ("pre2_fe_z", "%5.2f"), ("pre2_new_z", "%5.2f"), ("ucos", "%5.2f"), ("hmax", "%5.2f"),
        ("hmax_idx", "%4d"), ("h5", "%5.2f"), ("cand_maxcos", "%5.2f")]
hdr = " ".join(("%" + str(len(fmt % 0)) + "s") % c[:len(fmt % 0)] for c, fmt in show)
for k in pick:
    o, m = D.off[k], D.sz[k]
    sub = df[o:o + m].sort("rank")
    r = sub.row(0, named=True)
    print(f"\n### feed {r['fidx']} user {r['user_id']} M {m} H {r['H']} guest {r['guest']} uvec_miss {r['uvec_missing']} newest t/s/c {r['newest_type']}/{r['newest_src']}/{r['newest_cat']} ncats32 {r['n_hist_cats32']} nfeedcats {r['n_feed_cats']} hour {r['hour']} best {best[k]:.0f} cold_share {(sub['xp_imp_c50'] == 0).mean():.2f}")
    print(hdr)
    for row in sub.iter_rows(named=True):
        if row["rank"] <= 6 or row["label"] == 1:
            print(" ".join(fmt % (row[c] if row[c] is not None and row[c] == row[c] else -9) for c, fmt in show))
