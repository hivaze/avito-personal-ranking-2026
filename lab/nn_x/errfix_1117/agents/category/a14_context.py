"""Vertical x feed context (hour, weekday, guest, history length, feed size): calibration and NDCG cross-fit."""
import numpy as np
import polars as pl
import cu

SP = "<repo>/lab/nn_x/errfix_1117/agents/category/work"
C = cu.Ctx()
df = C.df
E = np.load(f"{SP}/E_flex.npy")
y = C.y.astype(float)
fid = C.fid
s = C.s
c = lambda n: df[n].to_numpy().astype(np.float64)
vert = np.array([cu.VERT[x] for x in df["cat"].to_list()])
hour = c("hour")
ctx = {
    "night0-6": (hour <= 6), "morning7-11": (hour >= 7) & (hour <= 11), "day12-17": (hour >= 12) & (hour <= 17),
    "evening18-23": hour >= 18, "wd1": c("weekday") == 1, "wd2": c("weekday") == 2, "wd3": c("weekday") == 3,
    "wd4": c("weekday") == 4, "guest": c("guest") == 1, "H<50": c("H") < 50, "H50-199": (c("H") >= 50) & (c("H") < 200),
    "H>=200": c("H") >= 200, "M<50": c("M") < 50, "M>=200": c("M") >= 200, "uvec_missing": c("uvec_missing") == 1,
}
print(df.group_by("weekday").agg(pl.col("fidx").n_unique()).sort("weekday"))
rows = []
nf = len(C.sz)
r = y - E
for k, m in ctx.items():
    for v in np.unique(vert):
        x = (m & (vert == v)).astype(float)
        if (E * x).sum() < 15:
            continue
        U_f = np.bincount(fid, weights=r * x, minlength=nf)
        z = U_f.sum() / np.sqrt((U_f ** 2).sum())
        rows.append(dict(ctx=k, vert=v, O=(y * x).sum(), E=round((E * x).sum(), 1), OE=round((y * x).sum() / (E * x).sum(), 3), z=round(z, 2)))
res = pl.DataFrame(rows).sort("z")
pl.Config.set_tbl_rows(40)
print(res.head(12))
print(res.tail(12))
print(len(rows), "cells")
# NDCG cross-fit for the strongest |z| cells
top = res.with_columns(az=pl.col("z").abs()).sort("az", descending=True).head(8)
grid = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]
for k, v in top.select("ctx", "vert").iter_rows():
    x = (ctx[k] & (vert == v)).astype(float)
    C.crossfit(grid, lambda w: s + w * x, label=f"{k} x {v}")
print("variants", C.n_tried)
