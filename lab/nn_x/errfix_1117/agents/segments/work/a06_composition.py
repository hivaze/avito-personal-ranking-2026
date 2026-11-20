"""Feed composition: does the model under-account for how many same-category competitors are in the feed?"""
import fx, kit, numpy as np, polars as pl, clogit
df = kit.load()
c = fx.Ctx(df)
y = c.y; feed = c.feed; s = c.base
M = df["M"].to_numpy().astype(float)
cn = df["cat_n_in_feed"].to_numpy().astype(float)
cr = df["cat_rank_in_feed"].to_numpy().astype(float)
share = cn / M
# baseline conditional logit with score only
a0, _, ll0 = clogit.fit(s, np.zeros((len(y), 0)), y, feed, c.nf)
print("score-only a", a0, "ll", ll0)
cands = {
    "log_cn": np.log(cn),
    "log_share": np.log(share),
    "log_cr": np.log(cr),
    "ndup95": df["n_cand_dup95"].to_numpy().astype(float),
    "cmx": np.nan_to_num(df["cand_maxcos"].to_numpy().astype(float), nan=0.8),
}
for nm, x in cands.items():
    a, w, ll = clogit.fit(s, x[:, None], y, feed, c.nf, l2=0.0)
    print(f"{nm:10s} a={a:.3f} w={w[0]:+.4f} dll={ll-ll0:.1f}")
# binned residual: y - expected under score-only clogit, by log_cn bins within rank buckets
z = a0 * s
zmax = np.full(c.nf, -np.inf); np.maximum.at(zmax, feed, z)
e = np.exp(z - zmax[feed]); se = np.bincount(feed, weights=e)
p = c.npos[feed] * e / se[feed]
rk = df["rank"].to_numpy()
T = pl.DataFrame({"y": y, "p": p, "rkb": np.digitize(rk, [2, 4, 11, 31, 101]), "cnb": np.digitize(cn, [2, 4, 10, 30, 80]),
                  "crb": np.digitize(cr, [2, 3, 6, 11, 31]), "Mb": np.digitize(M, [30, 100, 200])})
print(T.group_by("rkb", "cnb").agg(pl.len(), pl.col("y").sum().alias("pos"), (pl.col("y").sum() / pl.col("p").sum()).alias("obs/exp")).sort("rkb", "cnb").to_pandas().to_string())
print(T.group_by("rkb", "crb").agg(pl.len(), pl.col("y").sum().alias("pos"), (pl.col("y").sum() / pl.col("p").sum()).alias("obs/exp")).sort("rkb", "crb").to_pandas().to_string())
