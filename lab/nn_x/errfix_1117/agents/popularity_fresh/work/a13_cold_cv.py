"""Stability of cold & heavy-category boost: in-sample delta over thresholds and weights; per-fold."""
from lib import *
D = Data()
xp = D.col("xp_imp_c50"); cv = D.col("cat_views"); f4 = D.col("f4")
cold = (xp == 0).astype(float)
s = D.s
print("thr  " + " ".join("w=%.2f   " % w for w in [0.04, 0.08, 0.12, 0.2]))
for thr in [10, 20, 30, 40, 50, 60, 80, 100, 150]:
    row = []
    for w in [0.04, 0.08, 0.12, 0.2]:
        m, se = D.delta(s + w * cold * (cv >= thr) + 1e-7 * s)
        row.append("%+.5f" % m)
    print("%4d " % thr + " ".join(row), " rows", int((cold * (cv >= thr)).sum()))
# per-fold for thr 50, w .08
d = D.ndcg(s + 0.08 * cold * (cv >= 50) + 1e-7 * s) - D.base
print("per fold", [round(d[D.fold_feed == k].mean(), 5) for k in range(5)])
# what about cold only within head?
for thr in [30, 50]:
    for hd in [10, 30, 300]:
        m, se = D.delta(s + 0.08 * cold * (cv >= thr) * (D.col("rank") <= hd) + 1e-7 * s)
        print("thr", thr, "head", hd, "%+.5f (%.5f)" % (m, se))
# feed-level: what segment do the cv>=50 cold items live in?
fm = np.add.reduceat(cold * (cv >= 50), D.off) > 0
print("feeds with any cold & cv>=50:", fm.sum())
