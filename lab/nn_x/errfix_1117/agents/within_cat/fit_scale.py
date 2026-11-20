from wc import *
df = load(); d = features(df)
d = d.with_columns((pl.col("score") - pl.col("score").mean().over(["fidx", "cat"])).alias("s_dm"))
off, sz = kit.feed_layout(d)
y = d["label"].to_numpy().astype(np.float64); s = d["score"].to_numpy().astype(np.float64)
sdm = d["s_dm"].to_numpy().astype(np.float64)
lcn = np.log(d["cat_n_in_feed"].to_numpy().astype(np.float64))
scn = d["same_cat_newest"].to_numpy().astype(np.float64)
cu1 = (d["cat_user_rank"].to_numpy() == 1).astype(np.float64)
lM = np.log(d["M"].to_numpy().astype(np.float64))
for nm, cols in [("sdm,sdm*lcn", [sdm, sdm * lcn]), ("sdm,sdm*scn", [sdm, sdm * scn]), ("sdm,sdm*lcn,sdm*scn", [sdm, sdm * lcn, sdm * scn]),
                 ("sdm,sdm*lcn,sdm*scn,sdm*cu1", [sdm, sdm * lcn, sdm * scn, sdm * cu1]), ("sdm,sdm*lcn,sdm*lM", [sdm, sdm * lcn, sdm * lM]),
                 ("sdm,sdm*lcn,sdm*scn,sdm*lM", [sdm, sdm * lcn, sdm * scn, sdm * lM])]:
    X = np.c_[tuple(cols)]
    a, b, se, L = clogit(X, s, y, off, sz, l2=0)
    print("%-32s a=%.3f " % (nm, a) + " ".join("b%d=%+.4f(z%+.1f)" % (i, bi, bi / si) for i, (bi, si) in enumerate(zip(b, se))))
