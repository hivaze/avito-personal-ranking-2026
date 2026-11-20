"""Novelty hypothesis: in the user's heavily-browsed category, old/high-exposure items were likely seen before
(stale to this user); fresh/cold items are novel. A/E and lambda by xp x f4 inside segments of category familiarity."""
import lam as LM
from lib import *
D = Data(); F = len(D.sz)
L = LM.get(D, 1.0)
npos = np.add.reduceat(D.y, D.off)
p = softmax_p(D.s, D.off, D.sz, npos, 1.0)
xp = D.col("xp_imp_c50"); f4 = D.col("f4"); cv = D.col("cat_views"); cur = D.col("cat_user_rank"); rank = D.col("rank")
def cell(m):
    a, e = D.y[m].sum(), p[m].sum()
    pf = np.bincount(D.feed_of_row[m], weights=L[m], minlength=F)
    zl = pf.mean() / (pf.std(ddof=1) / np.sqrt(F))
    return f"{a / e:4.2f}/{(a - e) / np.sqrt(e):+4.1f}/{zl:+4.1f} n{m.sum():6d}"
xb = [("cold", xp == 0), ("xp<1.7", (xp > 0) & (xp < 1.7)), ("xp1.7-3.6", (xp >= 1.7) & (xp < 3.6)), ("xp>=3.6", xp >= 3.6)]
fb = [("f4<.2", f4 < .2), ("f4.2-.5", (f4 >= .2) & (f4 < .5)), ("f4.5-.8", (f4 >= .5) & (f4 < .8)), ("f4>=.8", f4 >= .8)]
for sname, sm in [("cat_user_rank==1", cur == 1), ("cat_user_rank 2-3", (cur >= 2) & (cur <= 3)), ("cat_user_rank>=4", cur >= 4), ("cat absent (0)", cur == 0),
                  ("cat_views>=50", cv >= 50), ("cat_views 10-49", (cv >= 10) & (cv < 50)), ("cat_views 1-9", (cv >= 1) & (cv < 10))]:
    print(f"\n== {sname}: rows {sm.sum()}  cells: A/E / z_ae / z_lambda")
    print("%-10s" % "" + "".join("%30s" % f for f, _ in fb))
    for xn, xm in xb:
        print("%-10s" % xn + "".join("%30s" % cell(sm & xm & fm) for _, fm in fb))
