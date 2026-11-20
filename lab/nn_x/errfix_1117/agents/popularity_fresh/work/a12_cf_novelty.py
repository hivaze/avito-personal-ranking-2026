"""Cross-fit: cold/fresh items tilted by the user's familiarity with their category (novelty hypothesis)."""
from lib import *
D = Data()
xp = D.col("xp_imp_c50"); f4 = D.col("f4"); cv = D.col("cat_views"); cur = D.col("cat_user_rank"); sh = D.col("cat_hist_share")
ca = D.col("cat_actions"); H = D.col("H")
cold = (xp == 0).astype(float)
young = (f4 < 0.2).astype(float)
s = D.s
NV = 0
def run(name, g, ws):
    global NV
    NV += len(ws)
    r = D.crossfit(lambda w: s + w * g + 1e-7 * s, ws)
    best = max(r["insample"], key=lambda t: t[1])
    print(f"{name:34s} CF {r['cf']:+.5f} se {r['se']:.5f} z {r['z']:5.2f} | in-sample best w={best[0]} {best[1]:+.5f} ({best[2]:.5f}) | chosen {r['chosen']}")
WP = [0.03, 0.05, 0.08, 0.12, 0.16, 0.2, 0.3]
WS = [-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3]
print("share mean over cold rows", sh[cold > 0].mean(), "log cv mean", np.log1p(cv[cold > 0]).mean())
run("cold & cur==1", cold * (cur == 1), WP)
run("cold & cv>=50", cold * (cv >= 50), WP)
run("cold & cv>=20", cold * (cv >= 20), WP)
run("cold * share", cold * sh, [0.1, 0.2, 0.3, 0.5, 0.8])
run("cold * (share-.25)", cold * (sh - 0.25), [0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
run("cold * (log1p(cv)-2)", cold * (np.log1p(cv) - 2), [0.01, 0.02, 0.04, 0.06, 0.1])
run("cold * 1[cur==0] (demote)", cold * (cur == 0), [-0.03, -0.05, -0.1, -0.15, -0.2, -0.3])
run("young & cur==0 (demote)", young * (cur == 0), [-0.03, -0.05, -0.1, -0.15, -0.2, -0.3])
run("young * (share-.25)", young * (sh - 0.25), [0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
run("cold*(cur==1) - cold*(cur==0)", cold * ((cur == 1).astype(float) - (cur == 0)), WP)
print("variants", NV)
