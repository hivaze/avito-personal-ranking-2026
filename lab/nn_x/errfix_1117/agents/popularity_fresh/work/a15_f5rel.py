"""f5 reliability: for heavily exposed items f5 is precise; does the model under-use it? Cross-fit tilts by f5 weighted by exposure."""
from lib import *
D = Data()
xp = D.col("xp_imp_c50"); f5 = D.col("f5"); s = D.s
sat = (xp >= 3.9).astype(float)
f5m = D.feedwise(f5, np.median)
NV = 0
def run(name, g, ws):
    global NV
    NV += len(ws)
    r = D.crossfit(lambda w: s + w * g + 1e-7 * s, ws)
    best = max(r["insample"], key=lambda t: t[1])
    print(f"{name:34s} CF {r['cf']:+.5f} se {r['se']:.5f} z {r['z']:5.2f} | in-sample best w={best[0]} {best[1]:+.5f} ({best[2]:.5f}) | chosen {r['chosen']}")
    return r
W = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
run("(f5-.45)*sat", (f5 - 0.45) * sat, W)
run("(f5-.45)*xp/3.93", (f5 - 0.45) * xp / 3.93, W)
run("(f5-.45)*(xp/3.93)^2", (f5 - 0.45) * (xp / 3.93) ** 2, W)
run("(f5-.45)*(xp/3.93)^4", (f5 - 0.45) * (xp / 3.93) ** 4, W)
run("(f5-f5med)*sat", (f5 - f5m) * sat, W)
run("(f5-f5med)*(xp/3.93)^2", (f5 - f5m) * (xp / 3.93) ** 2, W)
run("(f5-.45)*(xp>=3)", (f5 - 0.45) * (xp >= 3), W)
run("f5 (all)", f5 - 0.45, [-0.1, -0.05, 0.05, 0.1, 0.2])
print("variants", NV)
for w in [0.1, 0.2, 0.3, 0.5]:
    m, se = D.delta(s + w * (f5 - 0.45) * sat + 1e-7 * s)
    d = D.ndcg(s + w * (f5 - 0.45) * sat + 1e-7 * s) - D.base
    print("w", w, "%+.5f (%.5f)" % (m, se), "folds", [round(d[D.fold_feed == k].mean(), 5) for k in range(5)])
