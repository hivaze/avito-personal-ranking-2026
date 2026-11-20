"""NDCG cross-fit of category-level boosts: feed share of the category, history share, f3, within-category shrink."""
import numpy as np
import polars as pl
import cu

C = cu.Ctx()
df = C.df
s = C.s
c = lambda n: df[n].to_numpy().astype(np.float64)
M = c("M")
share = c("cat_n_in_feed") / M
lcn = np.log(c("cat_n_in_feed"))
hshare = c("cat_hist_share")
f3 = c("f3")
lcr = np.log(c("cat_rank_in_feed"))
# category mean score in feed
key = df.select((pl.col("fidx") * 100 + pl.col("cat")).alias("k"))["k"].to_numpy()
_, inv = np.unique(key, return_inverse=True)
cnt = np.bincount(inv)
cmean = np.bincount(inv, weights=s) / cnt
cmean = cmean[inv]
cmax = np.zeros(inv.max() + 1)
np.maximum.at(cmax, inv, s)
cmax = cmax[inv]
# within-feed standardised
fmean = np.bincount(C.fid, weights=share)[C.fid] / C.sz[C.fid]
ws = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
C.crossfit(ws, lambda w: s + w * share, label="share")
C.crossfit(ws, lambda w: s + w * 0.2 * lcn, label="0.2*log cat_n")
C.crossfit(ws, lambda w: s + w * hshare, label="hist share")
C.crossfit(ws, lambda w: s + w * f3, label="f3")
C.crossfit([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3], lambda a: s + a * (cmean - s), label="shrink to cat mean")
C.crossfit([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3], lambda w: s + w * lcr, label="+log cat_rank")
C.crossfit([0.0, 0.05, 0.1, 0.2, 0.3, 0.5], lambda w: s + w * cmean, label="+cat mean")
for MM in (50, 100, 150):
    C.crossfit(ws, lambda w: s + w * share * (M >= MM), label=f"share M>={MM}")
print("variants", C.n_tried)
