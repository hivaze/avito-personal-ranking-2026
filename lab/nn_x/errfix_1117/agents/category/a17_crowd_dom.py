"""(A) head-only crowd boost; (E) feature re-weighting inside the dominant category of large feeds."""
import numpy as np
import polars as pl
import cu

C = cu.Ctx()
df = C.df
s = C.s
c = lambda n: df[n].to_numpy().astype(np.float64)
rk = c("rank")
lcn = np.log(c("cat_n_in_feed"))
share = c("cat_n_in_feed") / c("M")
ws = [0.0, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2]
for K in (3, 5, 10, 30):
    C.crossfit(ws, lambda w: s + w * lcn * (rk <= K), label=f"log cat_n head K={K}")


def zs(x):
    m, sd = np.nanmean(x), np.nanstd(x)
    return np.nan_to_num((x - m) / sd)


feats = {"f4": zs(c("f4")), "f5": zs(c("f5")), "xp": zs(c("xp_imp_c50")), "lpop": zs(np.log1p(c("it_hist_all_luo"))),
         "hmax": zs(c("hmax")), "h5": zs(c("h5")), "ucos": zs(c("ucos")), "f1": zs(c("f1")), "f2": zs(c("f2")),
         "pre2new": zs(c("pre2_new_z")), "dft": c("s_ft") - c("s_to"), "cmaxcos": zs(c("cand_maxcos"))}
grid = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
dom = (share >= 0.5) & (c("M") >= 100)
print("dominant-category rows", dom.sum(), "feeds", len(np.unique(C.fid[dom])))
rows = []
for fn, x in feats.items():
    r = C.crossfit(grid, lambda w: s + w * x * dom, verbose=False)
    rows.append(dict(feat=fn, cf=r["cf"], se=r["se"], z=r["z"], ins=r["ins"], best=r["best"]))
print(pl.DataFrame(rows).sort("z", descending=True))
print("variants", C.n_tried)
