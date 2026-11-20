from wc import *
df = load(); d = features(df)
d = d.filter(pl.col("n_cand_dup95") >= 1)
d = d.with_columns(pl.len().over(["fidx", "cand_maxcos"]).alias("k"), pl.col("label").sum().over(["fidx", "cand_maxcos"]).alias("gp"))
d = d.filter((pl.col("k") >= 2) & (pl.col("gp") >= 1)).sort(["fidx", "cand_maxcos", "score"], descending=[False, False, True])
print("groups", d.select(pl.struct("fidx", "cand_maxcos").n_unique()).item(), "rows", d.height, "pos", d["label"].sum())
# how often is the positive the higher-scored member?
t = d.with_columns(pl.col("score").rank("ordinal", descending=True).over(["fidx", "cand_maxcos"]).alias("pr")).filter(pl.col("label") == 1)
print("positive position within pair-group:", t.group_by("pr").agg(pl.len()).sort("pr"))
both = d.group_by(["fidx", "cand_maxcos"]).agg(pl.col("label").sum().alias("s"), pl.len().alias("k")).filter(pl.col("k") == 2)
print("pairs both positive:", (both["s"] == 2).sum(), "of", both.height)
g = (d["fidx"].cast(pl.Float64) * 10 + d["cand_maxcos"].cast(pl.Float64)).to_numpy()
off = np.r_[0, np.nonzero(g[1:] != g[:-1])[0] + 1]; sz = np.diff(np.r_[off, len(g)])
y = d["label"].to_numpy().astype(np.float64); s = d["score"].to_numpy().astype(np.float64)
a0, _, _, L0 = clogit(np.zeros((len(s), 0)), s, y, off, sz)
print("scale", a0, "groups", len(sz))
rows = []
for c in BASEF + ["f2_home"]:
    X = zcols(d.with_columns(pl.col("f2_home").cast(pl.Float64)), [c])
    if np.allclose(X, X[0]):
        continue
    a, b, se, L = clogit(X, s, y, off, sz, l2=0)
    rows.append((c, b[0], b[0] / se[0], L0 - L))
rows.sort(key=lambda r: -abs(r[2]))
for r in rows[:25]: print("%-22s %+.4f z=%+.2f dLL=%.1f" % r)
