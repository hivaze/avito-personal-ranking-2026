"""Near-copy pairs inside the head: is the lower copy over-estimated (click split between copies)?
Fingerprint near-identical candidates by (cat, ucos, hmax, h1) closeness and cand_maxcos >= thr."""
import hk, numpy as np, polars as pl

c = hk.Ctx(); df = c.df
r = c.col("rank"); y = c.y
cat = c.col("cat"); uc = np.nan_to_num(c.col("ucos").astype(float), nan=-9); hm = np.nan_to_num(c.col("hmax").astype(float), nan=-9)
h1 = np.nan_to_num(c.col("h1").astype(float), nan=-9); cm = c.col("cand_maxcos").astype(float)
f2 = c.col("f2").astype(float)
K = 20
# expected rate by (M bucket, rank)
Mb = np.digitize(c.col("M"), [30, 60, 100, 200]); key = Mb * 1000 + np.minimum(r, 50)
uk, inv = np.unique(key, return_inverse=True); rate = (np.bincount(inv, weights=y) / np.bincount(inv))[inv]
lower_copy = np.zeros(len(df), bool); has_copy_above = np.zeros(len(df), bool)
pairs = []
for k, (o, m) in enumerate(zip(c.off, c.sz)):
    idx = o + np.nonzero(r[o:o + m] <= K)[0]
    idx = idx[np.argsort(r[idx])]
    for a_ in range(len(idx)):
        for b_ in range(a_ + 1, len(idx)):
            i, j = idx[a_], idx[b_]
            if cat[i] == cat[j] and cm[i] >= .97 and cm[j] >= .97 and abs(uc[i] - uc[j]) < .01 and abs(hm[i] - hm[j]) < .01 and abs(h1[i] - h1[j]) < .02:
                pairs.append((i, j))
                has_copy_above[j] = True
P = np.array(pairs)
print("near-copy pairs in top", K, ":", len(P), " feeds:", len(np.unique(c.feed_of_row[P[:, 0]])))
yi, yj = y[P[:, 0]], y[P[:, 1]]
print(" P(upper pos)", yi.mean().round(4), "expected", rate[P[:, 0]].mean().round(4))
print(" P(lower pos)", yj.mean().round(4), "expected", rate[P[:, 1]].mean().round(4))
print(" P(lower pos | upper pos)", yj[yi == 1].mean().round(3), " P(lower pos | upper neg)", yj[yi == 0].mean().round(3))
print(" same f2 share", (f2[P[:, 0]] == f2[P[:, 1]]).mean())
m = has_copy_above & (r <= K)
print("items with a copy above them in top", K, ":", m.sum(), "pos", y[m].mean().round(4), "expected", rate[m].mean().round(4))
# demotion rule test: subtract delta from items that have a near-copy above
for d in [0.05, 0.1, 0.2, 0.4]:
    new = c.score - d * has_copy_above
    c.rep(new, f"demote lower copy by {d}")
