"""Within-feed calibration by category and category state.

Expected positives per row = npos(feed) * softmax(a * score) within the feed (conditional logit, a fitted).
Obs/Exp by groups shows which groups the model under-ranks (O/E > 1) or over-ranks (O/E < 1) inside feeds.
"""
import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
import cu

df = cu.load()
fid, off, sz = cu.feed_id(df)
y = df["label"].to_numpy().astype(np.float64)
s = df["score"].to_numpy().astype(np.float64)
npos = np.bincount(fid, weights=y)


def softmax_within(z):
    mx = np.maximum.reduceat(z, off)
    e = np.exp(z - mx[fid])
    den = np.bincount(fid, weights=e)
    return e / den[fid]


def nll(a):
    p = softmax_within(a * s)
    return -(y * np.log(p + 1e-300)).sum()


r = minimize_scalar(nll, bounds=(0.1, 5), method="bounded")
a = r.x
print("conditional-logit slope a =", round(a, 4))
p = softmax_within(a * s)
E = npos[fid] * p
df = df.with_columns(pl.Series("E", E), pl.Series("vert", [cu.VERT.get(c, "?") for c in df["cat"].to_list()]),
                     pl.Series("catname", [cu.CATNAME.get(c, "?") for c in df["cat"].to_list()]))
df = df.with_columns(
    rb=pl.when(pl.col("rank") <= 3).then(pl.lit("01-03")).when(pl.col("rank") <= 10).then(pl.lit("04-10"))
    .when(pl.col("rank") <= 30).then(pl.lit("11-30")).when(pl.col("rank") <= 100).then(pl.lit("31-100"))
    .otherwise(pl.lit("101+")),
    cstate=pl.when(pl.col("cat_user_rank") == 0).then(pl.lit("absent"))
    .when(pl.col("cat_newest_idx") == 0).then(pl.lit("newest"))
    .when(pl.col("cat_newest_idx") < 5).then(pl.lit("idx1-4"))
    .when(pl.col("cat_newest_idx") < 32).then(pl.lit("idx5-31"))
    .otherwise(pl.lit("idx32+")),
)
df.write_parquet("<repo>/lab/nn_x/errfix_1117/agents/category/work/calib.parquet")


def oe(by, minE=20, sort="z"):
    g = (df.group_by(by).agg(n=pl.len(), O=pl.col("label").sum(), E=pl.col("E").sum(),
                             feeds=pl.col("fidx").n_unique())
         .with_columns(OE=pl.col("O") / pl.col("E"), z=(pl.col("O") - pl.col("E")) / pl.col("E").sqrt())
         .filter(pl.col("E") >= minE).sort(sort))
    return g


pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)
print(oe(["vert"]))
print(oe(["catname"], minE=15))
print(oe(["cstate"]))
print(oe(["rb"], sort="rb"))
print(oe(["cstate", "rb"], sort=["cstate", "rb"]))
print(oe(["vert", "cstate"], minE=15))
