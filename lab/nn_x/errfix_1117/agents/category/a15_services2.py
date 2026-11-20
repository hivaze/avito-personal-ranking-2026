"""Services demotion variants conditioned on the user's Services views."""
import numpy as np
import cu

C = cu.Ctx()
df = C.df
s = C.s
c = lambda n: df[n].to_numpy().astype(np.float64)
serv = df["cat"].to_numpy() == 50
v, a = c("cat_views"), c("cat_actions")
ws = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
r = {}
r["v0"] = C.crossfit(ws, lambda w: s - w * (serv & (v == 0)), label="Services & views==0")
r["v<=2"] = C.crossfit(ws, lambda w: s - w * (serv & (v <= 2)), label="Services & views<=2")
r["v0a>0"] = C.crossfit(ws, lambda w: s - w * (serv & (v == 0) & (a > 0)), label="Services & views==0 & actions>0")
r["decay"] = C.crossfit(ws, lambda w: s - w * serv / (1 + v), label="Services * 1/(1+views)")
r["all"] = C.crossfit(ws, lambda w: s - w * serv, label="Services all")
print("variants", C.n_tried)
# by feed segment for the v0 rule at its in-sample best
d = r["v0"]["D"][ws.index(r["v0"]["best"])]
M = c("M")[C.off]
for lo, hi in ((0, 50), (50, 150), (150, 400)):
    m = (M >= lo) & (M < hi)
    print(f"  M {lo}-{hi}: feeds {m.sum()} delta {d[m].mean():+.5f} changed {(d[m] != 0).mean():.3f}")
for k in range(5):
    print(f"  fold {k}: in-sample-best delta {d[C.ffold == k].mean():+.5f}")
