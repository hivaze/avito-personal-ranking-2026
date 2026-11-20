from wc import *
df = load(); d = features(df)
d = d.with_columns(((pl.col("xp_imp_c50") - 1.57) / 1.45 + (pl.col("lpop") - 1.6) / 1.5 + (pl.col("f4") - 0.5) / 0.275).alias("expo"))
d = rel(d, ["expo", "xp_imp_c50", "f4"])
s = d["score"].to_numpy().astype(np.float64); cn = d["cat_n_in_feed"].to_numpy()
print("expo sd", d["expo"].std(), "expo_dm sd", d["expo_dm"].std())
for nm in ["expo_dm", "xp_imp_c50_dm", "f4_dm"]:
    x = d[nm].to_numpy().astype(np.float64); x = x / x.std()
    grid = [(w, T) for w in [0.0, 0.01, 0.02, 0.04] for T in [20, 40]]
    r = cross_fit(d, lambda dd, p: s - p[0] * x * (cn >= p[1]) + 1e-7 * s, grid)
    print(nm, "oof %.5f se %.5f z %.2f" % (r["delta"], r["se"], r["z"]), r["chosen"], r["full"], flush=True)
