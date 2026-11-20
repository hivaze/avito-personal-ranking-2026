"""Category-specific offsets: Services when absent from history, ShiftGigs, and a cross-fitted table of
(vertical x category-state) offsets."""
import numpy as np
import polars as pl
import cu

C = cu.Ctx()
df = C.df
s = C.s
c = lambda n: df[n].to_numpy().astype(np.float64)
cat = df["cat"].to_numpy()
ur = c("cat_user_rank")
cni = c("cat_newest_idx")
ws = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
serv = cat == 50
C.crossfit(ws, lambda w: s - w * (serv & (ur == 0)), label="Services absent demote")
C.crossfit(ws, lambda w: s - w * serv, label="Services demote")
C.crossfit(ws, lambda w: s - w * (serv & ((ur == 0) | (ur >= 4))), label="Services ur0|ur>=4 demote")
C.crossfit(ws, lambda w: s - w * 0.1 * serv * np.where(ur == 0, 10, np.minimum(ur, 10)), label="Services * ur demote")
C.crossfit(ws, lambda w: s - w * (cat == 53), label="ShiftGigs demote")
C.crossfit(ws, lambda w: s - w * ((cat == 53) | (serv & (ur == 0))), label="ShiftGigs+Services-absent demote")
C.crossfit(ws, lambda w: s - w * (ur == 0), label="absent demote (all cats)")
for v in ("Hobby", "Animals", "Home", "Transport", "RealEstate", "Personal", "Electronics", "Jobs"):
    vm = np.array([cu.VERT[x] == v for x in cat.tolist()])
    C.crossfit([-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3], lambda w: s + w * (vm & (ur == 0)), label=f"{v} absent offset")
print("variants", C.n_tried)
