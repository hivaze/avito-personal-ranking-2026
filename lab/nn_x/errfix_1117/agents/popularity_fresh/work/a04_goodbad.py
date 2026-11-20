"""Compare good feeds (best positive rank 1-3) vs bad feeds (best positive 30+): lens features of
positives, of the top-3 negatives, overall; plus feed-level segment mix."""
from lib import *
D = Data(); df = D.df
rank = D.col("rank")
best = np.minimum.reduceat(np.where(D.y == 1, rank, 1e9), D.off)
bestr = best[D.feed_of_row]
good = bestr <= 3; bad = bestr >= 30
print("feeds good", (best <= 3).sum(), "bad", (best >= 30).sum(), "of", len(best))
cols = ["xp_imp_c50", "f4", "f5", "it_hist_all_luo", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "ucos", "hmax", "s_ft", "s_to"]
def summ(mask, name):
    out = [name.ljust(34), "%7d" % mask.sum()]
    for c in cols:
        x = D.col(c)[mask]
        if c == "xp_imp_c50":
            out.append("cold%.2f" % (x == 0).mean())
        out.append("%7.3f" % np.nanmedian(x))
    print(" ".join(out))
print(" ".join(["".ljust(34), "%7s" % "n"] + ["%7s" % c[:7] for c in ["cold"] + cols]))
summ(np.ones(len(D.y), bool), "all rows")
summ(good & (D.y == 1) & (rank <= 3), "good: positives at 1-3")
summ(good & (D.y == 0) & (rank <= 3), "good: negatives at 1-3")
summ(bad & (D.y == 1), "bad: positives (rank 30+)")
summ(bad & (D.y == 0) & (rank <= 3), "bad: negatives at 1-3")
summ(bad & (D.y == 0) & (rank > 30), "bad: negatives at 30+")
summ((D.y == 1) & (rank > 30), "all positives at 30+")
summ((D.y == 0) & (rank > 30), "all negatives at 30+")
summ((D.y == 1) & (rank <= 10), "all positives at 1-10")
summ((D.y == 0) & (rank <= 10), "all negatives at 1-10")
# feed-level mix
f = df.group_by("fidx", maintain_order=True).agg(pl.all().first())
fg = best <= 3; fb = best >= 30
for c in ["guest", "uvec_missing", "M", "H", "n_hist_cats32", "n_feed_cats", "hour"]:
    x = f[c].to_numpy().astype(float)
    print("%-14s good %.3f bad %.3f all %.3f" % (c, x[fg].mean(), x[fb].mean(), x.mean()))
xp = D.col("xp_imp_c50"); f4 = D.col("f4")
cs = np.add.reduceat((xp == 0).astype(float), D.off) / D.sz
fr = np.add.reduceat((f4 < 0.1).astype(float), D.off) / D.sz
print("cold share good %.3f bad %.3f all %.3f" % (cs[fg].mean(), cs[fb].mean(), cs.mean()))
print("fresh<.1 share good %.3f bad %.3f all %.3f" % (fr[fg].mean(), fr[fb].mean(), fr.mean()))
# positives' rank percentile by cold/fresh
pr = rank / D.col("M")
for name, m in [("cold pos", (xp == 0) & (D.y == 1)), ("warm pos", (xp > 0) & (D.y == 1)),
                ("fresh<.1 pos", (f4 < 0.1) & (D.y == 1)), ("old>.9 pos", (f4 > 0.9) & (D.y == 1))]:
    print("%-14s n %5d mean rank pct %.3f  share in top10 %.3f  share 30+ %.3f" % (name, m.sum(), pr[m].mean(), (rank[m] <= 10).mean(), (rank[m] > 30).mean()))
