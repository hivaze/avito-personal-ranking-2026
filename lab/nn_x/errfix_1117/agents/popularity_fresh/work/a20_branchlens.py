"""Item-level branch re-weighting by lens: score + w*(s_to - s_ft)*g(item)."""
from lib import *
D = Data()
xp = D.col("xp_imp_c50"); f4 = D.col("f4"); f5 = D.col("f5"); s = D.s
dd = D.col("s_to") - D.col("s_ft")
print("sd of s_to - s_ft", dd.std(), "cold", dd[xp == 0].std(), "warm", dd[xp > 0].std())
for name, g in [("cold", xp == 0), ("young f4<.2", f4 < .2), ("xp sat", xp >= 3.9), ("xp(0,1.7)", (xp > 0) & (xp < 1.7)),
                ("old f4>=.8", f4 >= .8), ("f5<.1", f5 < .1), ("all", np.ones(len(s), bool))]:
    g = g.astype(float)
    r = D.crossfit(lambda w: s + w * dd * g + 1e-7 * s, [-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3])
    best = max(r["insample"], key=lambda t: t[1])
    print("%-12s CF %+.5f se %.5f z %5.2f | best in-sample w=%s %+.5f (%.5f) chosen %s" % (name, r["cf"], r["se"], r["z"], best[0], best[1], best[2], r["chosen"]))
