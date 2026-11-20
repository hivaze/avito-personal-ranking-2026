from wc import *
df = load(); d = features(df)
s = d["score"].to_numpy().astype(np.float64)
rk = d["rank"].to_numpy()
ldup = d["ldup"].to_numpy().astype(np.float64)
dup1 = (d["n_cand_dup95"].to_numpy() >= 1).astype(np.float64)
cm = d["cand_maxcos"].fill_nan(0.85).to_numpy().astype(np.float64)
grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
for nm, x in [("ldup", ldup), ("dup1", dup1), ("cmc>.9", np.clip(cm - 0.9, 0, None) * 10)]:
    for K in [300, 30]:
        r = cross_fit(d, lambda dd, w: s + w * x * (rk <= K) + 1e-7 * s, grid)
        print(nm, "K", K, "oof %.5f se %.5f z %.2f" % (r["delta"], r["se"], r["z"]), r["chosen"], r["full"], flush=True)
