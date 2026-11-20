"""If an effect is real with log-odds beta (fitted by conditional logit), what NDCG gain should we expect?
Simulate labels from softmax(score + beta*g) per feed (npos positives drawn without replacement) and compare
rankings by score vs score + beta*g. Gives the expected gain of a correct calibration-type rule."""
import clogit
from lib import *
D = Data()
xp = D.col("xp_imp_c50"); cv = D.col("cat_views"); f4 = D.col("f4"); cur = D.col("cat_user_rank"); cat = D.col("cat"); f5 = D.col("f5")
cands = {
    "cold&cv>=50": ((xp == 0) & (cv >= 50)).astype(float),
    "cold&cur1": ((xp == 0) & (cur == 1)).astype(float),
    "fresh.1&cat48|50": ((f4 < .1) & np.isin(cat, [48, 50])).astype(float),
    "f5<.2&xp>3": ((f5 < .2) & (xp > 3)).astype(float),
}
npos = np.add.reduceat(D.y, D.off)
rng = np.random.default_rng(0)
for name, g in cands.items():
    b = clogit.fit(g[:, None], D.y, D.off, D.s, 1e-3, D.feed_of_row, len(D.sz))[0]
    # SE of b via observed information
    w1 = softmax_p(D.s + b * g, D.off, D.sz, np.ones(len(D.sz)))
    Eg = np.add.reduceat(w1 * g, D.off); Eg2 = np.add.reduceat(w1 * g * g, D.off)
    se = 1 / np.sqrt((npos * (Eg2 - Eg ** 2)).sum())
    new = D.s + b * g + 1e-7 * D.s
    gains = []
    for rep in range(20):
        ysim = np.zeros(len(D.s), np.int64)
        for o, m, n in zip(D.off, D.sz, npos):
            pp = np.exp(new[o:o + m] - new[o:o + m].max()); pp /= pp.sum()
            idx = rng.choice(m, size=min(n, m), replace=False, p=pp)
            ysim[o + idx] = 1
        from lib import _ndcg, DISC
        g0 = _ndcg(D.s, ysim, D.off, D.sz, DISC); g1 = _ndcg(new, ysim, D.off, D.sz, DISC)
        gains.append((g1 - g0).mean())
    print("%-18s beta %+.3f (se %.3f) -> expected NDCG gain if real: %+.5f (sim sd %.5f)" % (name, b, se, np.mean(gains), np.std(gains)))
