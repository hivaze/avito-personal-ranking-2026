"""Confident rank-1 (gap to rank 2 in the top quartile): right vs wrong. Which rank-1 attributes separate them?"""
from util import *

w = pl.read_parquet("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/head_wide.parquet")
q75 = w["g12"].quantile(0.75)
conf = w.filter(pl.col("g12") >= q75)
print("confident feeds", conf.height, "r1 precision", conf["label_1"].mean())
feats = [x for x in w.columns if x.endswith("_1") and x not in ("label_1",)]
out = []
for x in feats:
    a = conf.filter(pl.col("label_1") == 1)[x].cast(pl.Float64).fill_nan(None).drop_nulls().to_numpy()
    b = conf.filter(pl.col("label_1") == 0)[x].cast(pl.Float64).fill_nan(None).drop_nulls().to_numpy()
    if len(a) < 20 or len(b) < 20:
        continue
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = (a.mean() - b.mean()) / (se + 1e-12)
    out.append((abs(z), x, a.mean(), b.mean(), z))
for t in sorted(out, reverse=True)[:20]:
    print(f"  {t[1]:22s} right={t[2]:.4f} wrong={t[3]:.4f} z={t[4]:+.2f}")
for x in ["M", "n_hist_cats32", "H", "newest_type", "guest", "npos"]:
    a = conf.filter(pl.col("label_1") == 1)[x].cast(pl.Float64).to_numpy(); b = conf.filter(pl.col("label_1") == 0)[x].cast(pl.Float64).to_numpy()
    print(f"  feed {x:15s} right={a.mean():.3f} wrong={b.mean():.3f}")
# where is the positive in confident-wrong feeds? same category as r1?
df = load()
cw = conf.filter(pl.col("label_1") == 0).select("fidx", "cat_1")
p = df.filter(pl.col("label") == 1).join(cw, on="fidx").group_by("fidx").agg(
    pl.col("rank").min().alias("best"), (pl.col("cat") == pl.col("cat_1")).any().alias("pos_same_cat_as_r1"),
    pl.col("same_cat_newest").any().alias("pos_in_newest_cat"), (pl.col("cat_user_rank") == 0).any().alias("pos_cat_absent"))
print("confident-wrong feeds", p.height)
print(p.select(pl.col("best").median(), pl.col("pos_same_cat_as_r1").mean(), pl.col("pos_in_newest_cat").mean(), pl.col("pos_cat_absent").mean()))
cr = conf.filter(pl.col("label_1") == 1).select("fidx", "cat_1")
p2 = df.filter(pl.col("label") == 1).join(cr, on="fidx").group_by("fidx").agg((pl.col("cat") == pl.col("cat_1")).all().alias("all_same"))
print("confident-right feeds: all positives same cat as r1:", p2["all_same"].mean())
