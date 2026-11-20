from wc import *
df = load(); d = features(df)
d = d.sort(["fidx", "cat", "score"], descending=[False, False, True])
d = d.with_columns(pl.col("label").sum().over(["fidx", "cat"]).alias("gpos"))
d = d.filter((pl.col("gpos") > 0) & (pl.col("cat_n_in_feed") >= int(sys.argv[1]) if len(sys.argv) > 1 else pl.col("gpos") > 0))
d = rel(d, BASEF)
g = d["fc"].to_numpy()
off = np.r_[0, np.nonzero(g[1:] != g[:-1])[0] + 1]; sz = np.diff(np.r_[off, len(g)])
y = d["label"].to_numpy().astype(np.float64); s = d["score"].to_numpy().astype(np.float64)
print("groups", len(sz), "rows", len(y), "pos", y.sum())
a0, _, _, L0 = clogit(np.zeros((len(s), 0)), s, y, off, sz)
print("scale", a0)
rows = []
for c in BASEF:
    if c in ("lcatn", "cat_hist_share", "f3", "lcatnew"):
        continue
    X = zcols(d, [c])
    a, b, se, L = clogit(X, s, y, off, sz, l2=0)
    rows.append((c, b[0], b[0] / se[0], L0 - L))
rows.sort(key=lambda r: -abs(r[2]))
for r in rows: print("%-22s %+.4f z=%+.2f dLL=%.1f" % r)
