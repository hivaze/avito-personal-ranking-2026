"""Within-category spread per vertical: score + a * (catmean - score) (a>0 shrinks, a<0 expands)."""
import numpy as np
import polars as pl
import cu

C = cu.Ctx()
df = C.df
s = C.s
key = df.select((pl.col("fidx") * 100 + pl.col("cat")).alias("k"))["k"].to_numpy()
_, inv = np.unique(key, return_inverse=True)
cmean = (np.bincount(inv, weights=s) / np.bincount(inv))[inv]
cn = df["cat_n_in_feed"].to_numpy()
vert = np.array([cu.VERT[x] for x in df["cat"].to_list()])
grid = [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2]
rows = []
for v in np.unique(vert):
    m = (vert == v) & (cn >= 5)
    if m.sum() < 15000:
        continue
    r = C.crossfit(grid, lambda a: s + a * (cmean - s) * m, verbose=False)
    rows.append(dict(vert=v, cf=r["cf"], se=r["se"], z=r["z"], ins=r["ins"], best=r["best"], chosen=str(r["chosen"])))
for lo in (20, 50):
    m = cn >= lo
    r = C.crossfit(grid, lambda a: s + a * (cmean - s) * m, verbose=False)
    rows.append(dict(vert=f"all cat_n>={lo}", cf=r["cf"], se=r["se"], z=r["z"], ins=r["ins"], best=r["best"], chosen=str(r["chosen"])))
pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows).sort("z", descending=True))
print("variants", C.n_tried)
