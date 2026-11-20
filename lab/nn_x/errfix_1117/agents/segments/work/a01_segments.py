"""Step 1: per-feed table and NDCG / best-positive-rank by segment."""
import fx, kit, numpy as np, polars as pl
df = kit.load()
c = fx.Ctx(df)
off, sz, feed = c.off, c.sz, c.feed
y = c.y
rank = df["rank"].to_numpy()
best = np.full(c.nf, 10**6)
np.minimum.at(best, feed[y > 0], rank[y > 0])
s = c.base
# score shape per feed
def feed_top(x, k):
    out = np.zeros(c.nf)
    for i, (o, m) in enumerate(zip(off, sz)):
        xs = np.sort(x[o:o + m])[::-1]
        out[i] = xs[0] - xs[min(k, m - 1)]
    return out
g12 = feed_top(s, 1)
g15 = feed_top(s, 4)
g110 = feed_top(s, 9)
sft = df["s_ft"].to_numpy(); sto = df["s_to"].to_numpy()
corr = np.zeros(c.nf)
for i, (o, m) in enumerate(zip(off, sz)):
    corr[i] = np.corrcoef(sft[o:o + m], sto[o:o + m])[0, 1]
F = pl.DataFrame({
    "fidx": df["fidx"].to_numpy()[off], "user_id": df["user_id"].to_numpy()[off],
    "M": df["M"].to_numpy()[off], "H": df["H"].to_numpy()[off], "guest": df["guest"].to_numpy()[off],
    "uvec_missing": df["uvec_missing"].to_numpy()[off], "hour": df["hour"].to_numpy()[off],
    "weekday": df["weekday"].to_numpy()[off], "newest_type": df["newest_type"].to_numpy()[off],
    "newest_src": df["newest_src"].to_numpy()[off], "newest_cat": df["newest_cat"].to_numpy()[off],
    "n_hist_cats32": df["n_hist_cats32"].to_numpy()[off], "n_feed_cats": df["n_feed_cats"].to_numpy()[off],
    "npos": c.npos, "ndcg": c.base_nd, "best": best, "g12": g12, "g15": g15, "g110": g110, "corr_ft_to": corr,
    "fold": c.fold,
})
F.write_parquet("feeds.parquet")
print(F.describe())

def seg(name, key):
    t = F.with_columns(pl.Series("k", key)).group_by("k").agg(
        pl.len().alias("n"), pl.col("ndcg").mean().alias("ndcg"), (pl.col("best") <= 3).mean().alias("top3"),
        (pl.col("best") > 30).mean().alias("b30"), pl.col("npos").mean().alias("npos"), pl.col("M").mean().alias("M")
    ).sort("k")
    print("==", name); print(t)

Fm = F["M"].to_numpy(); Fh = F["H"].to_numpy()
seg("M", np.digitize(Fm, [20, 50, 100, 150, 200, 250, 300]))
seg("H", np.digitize(Fh, [1, 5, 20, 50, 100, 200, 256]))
seg("guest", F["guest"].to_numpy())
seg("uvec_missing", F["uvec_missing"].to_numpy())
seg("hour", np.digitize(F["hour"].to_numpy(), [3, 7, 10, 13, 16, 19, 22]))
seg("weekday", F["weekday"].to_numpy())
seg("newest_type", F["newest_type"].to_numpy())
seg("newest_src", F["newest_src"].to_numpy())
seg("n_feed_cats", np.digitize(F["n_feed_cats"].to_numpy(), [3, 6, 10, 15, 20, 30]))
seg("n_hist_cats32", np.digitize(F["n_hist_cats32"].to_numpy(), [1, 2, 3, 5, 8, 12]))
seg("g12 q", np.digitize(g12, np.quantile(g12, [.2, .4, .6, .8])))
seg("g110 q", np.digitize(g110, np.quantile(g110, [.2, .4, .6, .8])))
seg("corr ft/to q", np.digitize(corr, np.quantile(corr, [.2, .4, .6, .8])))
