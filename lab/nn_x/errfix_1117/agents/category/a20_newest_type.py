"""same-category-as-newest boost conditioned on the newest event's type / source / vertical."""
import numpy as np
import polars as pl
import cu

C = cu.Ctx()
df = C.df
s = C.s
c = lambda n: df[n].to_numpy().astype(np.float64)
same = c("same_cat_newest")
nt, ns = c("newest_type"), c("newest_src")
nvert = np.array([cu.VERT.get(x, "none") for x in df["newest_cat"].to_list()])
ws = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]
segs = {"nt=1": nt == 1, "nt in 2,3": np.isin(nt, (2, 3)), "nt=4": nt == 4, "nt in 5,6,8": np.isin(nt, (5, 6, 8)),
        "nt=7": nt == 7, "nt=10": nt == 10, "ns=0": ns == 0, "ns=1": ns == 1, "ns=2": ns == 2, "ns in 3,4,5,6,7": np.isin(ns, (3, 4, 5, 6, 7))}
for v in ("Services", "RealEstate", "Jobs", "Transport", "Personal", "Home", "Electronics", "Hobby"):
    segs[f"nvert={v}"] = nvert == v
rows = []
for k, m in segs.items():
    r = C.crossfit(ws, lambda w: s + w * same * m, verbose=False)
    rows.append(dict(seg=k, feeds=len(np.unique(C.fid[m])), cf=r["cf"], se=r["se"], z=r["z"], ins=r["ins"], best=r["best"], chosen=str(r["chosen"])))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows).sort("z", descending=True))
print("variants", C.n_tried)
