from wc import *
df = load(); d = features(df)
d = rel(d, ["f2"])
s = d["score"].to_numpy().astype(np.float64)
for nm in ["f2", "f2_dm"]:
    x = d[nm].to_numpy().astype(np.float64)
    r = cross_fit(d, lambda dd, w: s - w * x + 1e-7 * s, [0.0, 0.02, 0.04, 0.07, 0.1])
    print(nm, "oof %.5f se %.5f z %.2f" % (r["delta"], r["se"], r["z"]), r["chosen"], r["full"])
