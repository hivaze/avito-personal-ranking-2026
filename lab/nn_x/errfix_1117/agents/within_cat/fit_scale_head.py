from wc import *
df = load(); d = features(df)
d = d.with_columns((pl.col("score") - pl.col("score").mean().over(["fidx", "cat"])).alias("s_dm"))
for K in [10, 20, 50]:
    h = d.filter(pl.col("rank") <= K).sort(["fidx", "rank"])
    h = h.with_columns(pl.col("label").sum().over("fidx").alias("hp")).filter(pl.col("hp") > 0)
    off, sz = kit.feed_layout(h)
    y = h["label"].to_numpy().astype(np.float64); s = h["score"].to_numpy().astype(np.float64)
    sdm = h["s_dm"].to_numpy().astype(np.float64)
    lcn = np.log(h["cat_n_in_feed"].to_numpy().astype(np.float64))
    scn = h["same_cat_newest"].to_numpy().astype(np.float64)
    for nm, cols in [("sdm,sdm*lcn", [sdm, sdm * lcn]), ("sdm,sdm*scn", [sdm, sdm * scn]), ("lcn", [lcn])]:
        X = np.c_[tuple(cols)]
        a, b, se, L = clogit(X, s, y, off, sz, l2=0)
        print("K=%d %-14s a=%.3f " % (K, nm, a) + " ".join("b%d=%+.4f(z%+.1f)" % (i, bi, bi / si) for i, (bi, si) in enumerate(zip(b, se))))
