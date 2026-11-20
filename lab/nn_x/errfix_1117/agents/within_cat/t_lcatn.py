from wc import *
df = load(); d = features(df)
s = d["score"].to_numpy().astype(np.float64)
for col in ["lcatn", "cat_hist_share", "f3"]:
    x = d[col].to_numpy().astype(np.float64)
    grid = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15]
    r = cross_fit(d, lambda dd, w: s + w * x + 1e-7 * s, grid)
    print(col, "oof %.5f se %.5f z %.2f" % (r["delta"], r["se"], r["z"]), r["chosen"], r["full"])
