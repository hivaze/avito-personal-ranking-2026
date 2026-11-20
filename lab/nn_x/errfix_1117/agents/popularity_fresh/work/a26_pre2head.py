"""Head items the encoder dislikes (pre2 z < 0 while ranked top-k): demote? cross-fit."""
from lib import *
D = Data(); s = D.s; rank = D.col("rank")
pfe = D.col("pre2_fe_z"); pnew = D.col("pre2_new_z")
npos = np.add.reduceat(D.y, D.off); p = softmax_p(D.s, D.off, D.sz, npos, 1.0)
for k in [1, 3, 5]:
    for nm, m in [("fe<0", pfe < 0), ("both<0", (pfe < 0) & (pnew < 0)), ("both<-.5", (pfe < -.5) & (pnew < -.5))]:
        mm = m & (rank <= k)
        a, e = D.y[mm].sum(), p[mm].sum()
        g = mm.astype(float)
        r = D.crossfit(lambda w: s + w * g + 1e-7 * s, [-0.05, -0.1, -0.2, -0.3, -0.5])
        best = max(r["insample"], key=lambda t: t[1])
        print("top%d %-9s n %5d A/E %.3f | CF %+.5f se %.5f z %5.2f | best in-sample w=%s %+.5f" % (k, nm, mm.sum(), a / e, r["cf"], r["se"], r["z"], best[0], best[1]))
