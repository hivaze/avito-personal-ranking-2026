from wc import *
df = load(); d = features(df)
a = d["s_ft"].to_numpy().astype(np.float64); b = d["s_to"].to_numpy().astype(np.float64)
s = d["score"].to_numpy().astype(np.float64)
cu = d["cat_user_rank"].to_numpy()
print("corr s_ft s_to", np.corrcoef(a, b)[0, 1], "cu0 rows share", (cu == 0).mean())
grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for nm, mask in [("cu0", cu == 0), ("cu0|cu>=5", (cu == 0) | (cu >= 5))]:
    r = cross_fit(d, lambda dd, lam: np.where(mask, (1 - lam) * a + lam * b, s), grid)
    print(nm, "oof %.5f se %.5f z %.2f" % (r["delta"], r["se"], r["z"]), r["chosen"], r["full"])
# likelihood view: clogit with s and (s_to - s_ft)*1[cu0]
off, sz = kit.feed_layout(d); y = d["label"].to_numpy().astype(float)
for nm, mask in [("cu0", cu == 0)]:
    X = ((b - a) * mask)[:, None]
    aa, bb, se, L = clogit(X, s, y, off, sz, l2=0)
    print("clogit (s_to-s_ft)*cu0:", bb, bb / se)
