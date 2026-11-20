"""Score tests for continuous category-state features and slope/branch interactions (flexible baseline)."""
import numpy as np
import polars as pl
import cu

SP = "<repo>/lab/nn_x/errfix_1117/agents/category/work"
df = pl.read_parquet(f"{SP}/calib.parquet")
E = np.load(f"{SP}/E_flex.npy")
fid, off, sz = cu.feed_id(df)
nf = len(sz)
y = df["label"].to_numpy().astype(np.float64)
r = y - E


def test(x, name):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x)
    U_f = np.bincount(fid, weights=r * x, minlength=nf)
    se = np.sqrt((U_f ** 2).sum())
    return dict(name=name, z=round(U_f.sum() / se if se > 0 else 0, 2))


c = lambda n: df[n].to_numpy().astype(np.float64)
s, sft, sto = c("score"), c("s_ft"), c("s_to")
dft = sft - sto
vert = df["vert"].to_numpy()
cat = df["cat"].to_numpy()
cst = df["cstate"].to_numpy()
rk = c("rank")
feats = {
    "log_cat_views": np.log1p(c("cat_views")), "log_cat_actions": np.log1p(c("cat_actions")),
    "cat_newest_idx_log": np.log1p(np.where(c("cat_newest_idx") < 0, 300, c("cat_newest_idx"))),
    "cat_user_rank": np.where(c("cat_user_rank") == 0, 30, c("cat_user_rank")),
    "cat_hist_share": c("cat_hist_share"), "same_cat_newest": c("same_cat_newest"),
    "log_cat_n_in_feed": np.log(c("cat_n_in_feed")), "cat_share_in_feed": c("cat_n_in_feed") / c("M"),
    "log_cat_rank_in_feed": np.log(c("cat_rank_in_feed")), "cat_rank1": (c("cat_rank_in_feed") == 1),
    "f3": c("f3"), "f1": c("f1"), "f2_home": c("f2_home"), "f4": c("f4"), "f5": c("f5"),
    "act_ratio": c("cat_actions") / (c("cat_views") + c("cat_actions") + 1),
    "dft": dft,
}
rows = []
for k, v in feats.items():
    rows.append(test(v, k))
    for vv in np.unique(vert):
        m = vert == vv
        if m.sum() > 20000:
            rows.append(test(v * m, f"{k} x vert:{vv}"))
for vv in np.unique(vert):
    m = vert == vv
    rows.append(test(s * m, f"score x vert:{vv}"))
    rows.append(test(dft * m, f"dft x vert:{vv}"))
for cc in np.unique(cst):
    m = cst == cc
    rows.append(test(s * m, f"score x cstate:{cc}"))
    rows.append(test(dft * m, f"dft x cstate:{cc}"))
for nt in (1, 2, 3, 4):
    m = (c("newest_type") == nt)
    rows.append(test(dft * m, f"dft x newest_type:{nt}"))
for ns in (0, 1, 2):
    m = (c("newest_src") == ns)
    rows.append(test(dft * m, f"dft x newest_src:{ns}"))
for g in (0, 1):
    m = c("guest") == g
    rows.append(test(dft * m, f"dft x guest:{g}"))
for rb in ("01-03", "04-10", "11-30", "31-100", "101+"):
    m = df["rb"].to_numpy() == rb
    rows.append(test(dft * m, f"dft x rb:{rb}"))
res = pl.DataFrame(rows).sort("z")
pl.Config.set_tbl_rows(400)
pl.Config.set_tbl_width_chars(200)
print(res.filter(pl.col("z").abs() >= 1.5))
print(len(rows), "tests")
