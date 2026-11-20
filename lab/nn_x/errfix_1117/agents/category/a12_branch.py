"""Branch re-weighting (s_ft vs s_to) by feed segment defined by category context."""
import numpy as np
import polars as pl
import cu

C = cu.Ctx()
df = C.df
s = C.s
c = lambda n: df[n].to_numpy().astype(np.float64)
dft = c("s_ft") - c("s_to")
grid = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]
C.crossfit(grid, lambda w: s + w * dft, label="global dft")
nc = df["newest_cat"].to_numpy()
nvert = np.array([cu.VERT.get(x, "none") for x in nc.tolist()])
segs = {}
for v in np.unique(nvert):
    if (nvert == v).sum() > 20000:
        segs[f"newest_vert:{v}"] = nvert == v
nhc = c("n_hist_cats32")
for lo, hi in ((0, 2), (3, 4), (5, 6), (7, 99)):
    segs[f"nhc32:{lo}-{hi}"] = (nhc >= lo) & (nhc <= hi)
nfc = c("n_feed_cats")
for lo, hi in ((1, 10), (11, 15), (16, 20), (21, 99)):
    segs[f"nfeedcats:{lo}-{hi}"] = (nfc >= lo) & (nfc <= hi)
for v in (0, 1, 2):
    segs[f"newest_src:{v}"] = c("newest_src") == v
for v in (1, 2, 3, 4):
    segs[f"newest_type:{v}"] = c("newest_type") == v
segs["guest"] = c("guest") == 1
same = c("same_cat_newest") == 1
segs["item same_cat_newest"] = same
segs["item absent cat"] = c("cat_user_rank") == 0
rows = []
for k, m in segs.items():
    r = C.crossfit(grid, lambda w: s + w * dft * m, verbose=False)
    rows.append(dict(seg=k, n=int(m.sum()), cf=r["cf"], se=r["se"], z=r["z"], ins=r["ins"], best=r["best"], chosen=str(r["chosen"])))
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows).sort("z", descending=True))
print("variants", C.n_tried)
