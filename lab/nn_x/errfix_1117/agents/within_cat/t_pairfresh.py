from wc import *
df = load(); d = features(df)
d = d.with_columns(pl.when(pl.col("n_cand_dup95") >= 1).then(pl.len().over(["fidx", "cand_maxcos"])).otherwise(0).alias("k"))
d = d.with_columns(pl.when(pl.col("k") == 2).then(
    pl.when(pl.col("f4") < pl.col("f4").max().over(["fidx", "cand_maxcos"])).then(1.0)
      .when(pl.col("f4") > pl.col("f4").min().over(["fidx", "cand_maxcos"])).then(-1.0).otherwise(0.0)).otherwise(0.0).alias("fresher"))
s = d["score"].to_numpy().astype(np.float64); x = d["fresher"].to_numpy()
print("rows with fresher!=0:", (x != 0).sum())
r = cross_fit(d, lambda dd, w: s + w * x, [0.0, 0.02, 0.05, 0.1, 0.2])
print("pair fresher: oof %.5f se %.5f z %.2f" % (r["delta"], r["se"], r["z"]), r["chosen"], r["full"])
