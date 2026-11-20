"""Head duels: adjacent pairs (rank r, r+1), r <= R, exactly one of them positive.
Outcome = 1 if the lower-ranked item is the positive. Which category-feature differences predict an upset,
beyond the score gap? Clustered (feed) z for each feature in a logistic model with the gap."""
import numpy as np
import polars as pl
import cu

C = cu.Ctx()
df = C.df.sort(["fidx", "rank"])
fid_sorted = df["fidx"].to_numpy()
rk = df["rank"].to_numpy()
y = df["label"].to_numpy()
c = lambda n: df[n].to_numpy().astype(np.float64)
cat = df["cat"].to_numpy()
cni = c("cat_newest_idx")
ur = c("cat_user_rank")
vert = np.array([cu.VERT[x] for x in cat.tolist()])
F = {
    "same_cat_newest": c("same_cat_newest"), "absent": (ur == 0).astype(float),
    "log_cni": np.log1p(np.where(cni < 0, 300, cni)), "log_views": np.log1p(c("cat_views")),
    "log_actions": np.log1p(c("cat_actions")), "hshare": c("cat_hist_share"), "lcat_n": np.log(c("cat_n_in_feed")),
    "cat_rank1": (c("cat_rank_in_feed") == 1).astype(float), "lcat_rank": np.log(c("cat_rank_in_feed")),
    "f3": c("f3"), "f1": c("f1"), "f4": c("f4"), "f5": c("f5"), "xp": c("xp_imp_c50"),
    "lpop": np.log1p(c("it_hist_all_luo")), "hmax": c("hmax"), "h1": c("h1"), "h5": c("h5"), "ucos": c("ucos"),
    "dft": c("s_ft") - c("s_to"), "cmaxcos": c("cand_maxcos"), "ndup": np.log1p(c("n_cand_dup95")),
    "services": (cat == 50).astype(float), "jobs": (vert == "Jobs").astype(float),
    "realestate": (vert == "RealEstate").astype(float), "personal": (vert == "Personal").astype(float),
    "transport": (vert == "Transport").astype(float), "home": (vert == "Home").astype(float),
    "electronics": (vert == "Electronics").astype(float), "hobby": (vert == "Hobby").astype(float),
}
s = c("score")
for R in (3, 10):
    i = np.nonzero((rk[:-1] <= R) & (fid_sorted[:-1] == fid_sorted[1:]))[0]
    j = i + 1
    one = (y[i] + y[j]) == 1
    i, j = i[one], j[one]
    out = y[j].astype(float)
    gap = s[i] - s[j]
    groups = fid_sorted[i]
    sameq = (cat[i] == cat[j])
    print(f"R={R}: duels {len(i)}, upset rate {out.mean():.3f}, same-category duels {sameq.mean():.3f}")
    rows = []
    for k, x in F.items():
        dx = np.nan_to_num(x[j] - x[i])
        if np.std(dx) == 0:
            continue
        X = np.column_stack([np.ones(len(gap)), gap, dx])
        b, se = cu.logit_cluster(X, out, groups)
        rows.append(dict(feat=k, coef=b[2], z=b[2] / se[2], frac_nonzero=(dx != 0).mean()))
    print(pl.DataFrame(rows).sort("z"))
