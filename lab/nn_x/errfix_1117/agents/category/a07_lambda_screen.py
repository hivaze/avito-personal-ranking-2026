"""NDCG-aligned screen: LambdaRank directional gradient z for category features."""
import numpy as np
import polars as pl
import cu

C = cu.Ctx()
df = C.df
pairs = cu.lambda_pairs(C, temp=1.0)
print("pairs", len(pairs[0]))
c = lambda n: df[n].to_numpy().astype(np.float64)
s, dft = C.s, c("s_ft") - c("s_to")
vert = np.array([cu.VERT[x] for x in df["cat"].to_list()])
cat = df["cat"].to_numpy()
cni = c("cat_newest_idx")
cst = np.where(c("cat_user_rank") == 0, "absent", np.where(cni == 0, "newest", np.where(cni < 5, "idx1-4", np.where(cni < 32, "idx5-31", "idx32+"))))
rk = c("rank")
feats = {
    "log_cat_views": np.log1p(c("cat_views")), "log_cat_actions": np.log1p(c("cat_actions")),
    "cat_newest_idx_log": np.log1p(np.where(cni < 0, 300, cni)),
    "cat_user_rank": np.where(c("cat_user_rank") == 0, 30, c("cat_user_rank")),
    "cat_hist_share": c("cat_hist_share"), "same_cat_newest": c("same_cat_newest"),
    "log_cat_n_in_feed": np.log(c("cat_n_in_feed")), "cat_share_in_feed": c("cat_n_in_feed") / c("M"),
    "log_cat_rank_in_feed": np.log(c("cat_rank_in_feed")), "cat_rank1": (c("cat_rank_in_feed") == 1),
    "f3": c("f3"), "f1": c("f1"), "f2_home": c("f2_home"), "f4": c("f4"), "f5": c("f5"),
    "act_ratio": c("cat_actions") / (c("cat_views") + c("cat_actions") + 1), "dft": dft,
}
rows = []
def add(x, name):
    z, m = cu.lambda_test(C, pairs, x)
    rows.append(dict(name=name, z=round(z, 2), g=m))
for k, v in feats.items():
    add(v, k)
    for vv in np.unique(vert):
        m = vert == vv
        if m.sum() > 20000:
            add(v * m, f"{k} x vert:{vv}")
for vv in np.unique(vert):
    add(vert == vv, f"vert:{vv}")
    add(s * (vert == vv), f"score x vert:{vv}")
    add(dft * (vert == vv), f"dft x vert:{vv}")
    for cc in np.unique(cst):
        add((vert == vv) & (cst == cc), f"vert:{vv} x cst:{cc}")
for cc in np.unique(cat):
    add(cat == cc, f"cat:{cu.CATNAME[cc]}")
for cc in np.unique(cst):
    m = cst == cc
    add(m, f"cst:{cc}")
    add(s * m, f"score x cst:{cc}")
    add(dft * m, f"dft x cst:{cc}")
for nt in (1, 2, 3, 4):
    add(dft * (c("newest_type") == nt), f"dft x newest_type:{nt}")
    add(c("same_cat_newest") * (c("newest_type") == nt), f"same x newest_type:{nt}")
for ns in (0, 1, 2):
    add(dft * (c("newest_src") == ns), f"dft x newest_src:{ns}")
    add(c("same_cat_newest") * (c("newest_src") == ns), f"same x newest_src:{ns}")
for g in (0, 1):
    add(dft * (c("guest") == g), f"dft x guest:{g}")
res = pl.DataFrame(rows).sort("z")
pl.Config.set_tbl_rows(400)
pl.Config.set_tbl_width_chars(200)
print(res.filter(pl.col("z").abs() >= 1.8))
print(len(rows), "tests")
res.write_csv("<repo>/lab/nn_x/errfix_1117/agents/category/work/lambda1.csv")
