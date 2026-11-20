from wc import *
df = load(); d = features(df)
d = d.with_columns((pl.col("score") - pl.col("score").mean().over(["fidx", "cat"])).alias("s_dm"))
s = d["score"].to_numpy().astype(np.float64)
sdm = d["s_dm"].to_numpy().astype(np.float64)
lcn = np.log(d["cat_n_in_feed"].to_numpy().astype(np.float64))
scn = d["same_cat_newest"].to_numpy().astype(np.float64)
grid = [(w, c) for w in [0.0, 0.03, 0.06, 0.09, 0.12] for c in [2.0, 2.9, 3.6]]
r = cross_fit(d, lambda dd, p: s + p[0] * (lcn - p[1]) * sdm, grid)
print("stretch lcn: oof %.5f se %.5f z %.2f" % (r["delta"], r["se"], r["z"]), r["chosen"])
for k, v in r["full"].items(): print("   ", k, v)
grid2 = [0.0, 0.05, 0.1, 0.15, 0.2]
r = cross_fit(d, lambda dd, w: s + w * (scn - 0.5) * sdm, grid2)
print("stretch scn: oof %.5f se %.5f z %.2f" % (r["delta"], r["se"], r["z"]), r["chosen"], r["full"])
