"""Brute-force NDCG cross-fit: per-vertical (and per-category-state) re-weighting of single features.
Rule: s + w * zx * [segment], zx = feature standardised over all rows (NaN -> 0)."""
import numpy as np
import polars as pl
import cu

C = cu.Ctx()
df = C.df
s = C.s
c = lambda n: df[n].to_numpy().astype(np.float64)
cat = df["cat"].to_numpy()
vert = np.array([cu.VERT[x] for x in cat.tolist()])
cni = c("cat_newest_idx")
ur = c("cat_user_rank")


def zs(x):
    x = np.where(np.isfinite(x), x, np.nan)
    m, sd = np.nanmean(x), np.nanstd(x)
    return np.nan_to_num((x - m) / sd)


feats = {
    "f4": zs(c("f4")), "f5": zs(c("f5")), "xp": zs(c("xp_imp_c50")), "lpop": zs(np.log1p(c("it_hist_all_luo"))),
    "hmax": zs(c("hmax")), "h5": zs(c("h5")), "ucos": zs(c("ucos")), "f1": zs(c("f1")), "f2": zs(c("f2")),
    "f2home": zs(c("f2_home")), "seqcv": zs(np.log1p(c("seq_cv_invd20"))), "pre2fe": zs(c("pre2_fe_z")),
    "pre2new": zs(c("pre2_new_z")), "dft": c("s_ft") - c("s_to"), "cmaxcos": zs(c("cand_maxcos")),
    "lcrank": zs(np.log(c("cat_rank_in_feed"))),
}
segs = {f"vert:{v}": vert == v for v in np.unique(vert) if (vert == v).sum() > 15000}
segs.update({"cst:newest": cni == 0, "cst:absent": ur == 0, "cst:idx1-4": (cni >= 1) & (cni < 5),
             "cst:idx5+": (cni >= 5)})
grid = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
rows = []
for sn, m in segs.items():
    mf = m.astype(float)
    for fn, x in feats.items():
        r = C.crossfit(grid, lambda w: s + w * x * mf, verbose=False)
        rows.append(dict(seg=sn, feat=fn, cf=r["cf"], se=r["se"], z=r["z"], ins=r["ins"], best=r["best"],
                         chosen=str(r["chosen"])))
res = pl.DataFrame(rows).sort("z", descending=True)
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
print(res.head(30))
print(res.tail(5))
print("families", len(rows), "variants", C.n_tried)
res.write_csv("<repo>/lab/nn_x/errfix_1117/agents/category/a11_results.csv")
