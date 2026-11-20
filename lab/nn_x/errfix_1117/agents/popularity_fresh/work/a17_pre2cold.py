"""Encoder scores for cold vs warm items: is pre2 over/under-trusted when the item has no interactions?"""
from lib import *
D = Data()
npos = np.add.reduceat(D.y, D.off)
p = softmax_p(D.s, D.off, D.sz, npos, 1.0)
xp = D.col("xp_imp_c50"); f4 = D.col("f4")
for c in ["pre2_fe_z", "pre2_new_z"]:
    x = D.col(c)
    print(f"\n== {c}")
    for gn, gm in [("cold f4<.2", (xp == 0) & (f4 < .2)), ("cold f4>=.2", (xp == 0) & (f4 >= .2)), ("xp(0,1.7)", (xp > 0) & (xp < 1.7)), ("xp>=1.7", xp >= 1.7)]:
        line = "%-12s" % gn
        for lo, hi in [(-9, -1), (-1, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 9)]:
            m = gm & (x >= lo) & (x < hi)
            a, e = D.y[m].sum(), p[m].sum()
            line += " | [%g,%g) %.2f(%+4.1f)" % (lo, hi, a / e, (a - e) / np.sqrt(e))
        print(line)
    # corr between pre2 and score among cold vs warm
    for gn, gm in [("cold", xp == 0), ("warm", xp > 0)]:
        print("  corr(score, %s) %s: %.3f" % (c, gn, np.corrcoef(D.s[gm], x[gm])[0, 1]))
s = D.s
for c in ["pre2_fe_z", "pre2_new_z"]:
    x = D.col(c)
    for gn, g in [("cold", (xp == 0).astype(float)), ("cold&young", ((xp == 0) & (f4 < .2)).astype(float)), ("warm", (xp > 0).astype(float))]:
        r = D.crossfit(lambda w: s + w * g * x + 1e-7 * s, [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1])
        best = max(r["insample"], key=lambda t: t[1])
        print("%s x %-10s CF %+.5f se %.5f z %.2f | best in-sample w=%s %+.5f" % (c, gn, r["cf"], r["se"], r["z"], best[0], best[1]))
