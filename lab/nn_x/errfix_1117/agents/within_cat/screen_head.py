"""Conditional logit restricted to the head (top-K of each feed): which features carry residual signal there?"""
from wc import *
K = int(sys.argv[1])
df = load(); d = features(df)
d = rel(d, BASEF)
d = d.with_columns((pl.col("cat") == pl.col("cat").first().over("fidx", order_by="rank")).alias("same_as_top"),
                   pl.len().over(["fidx", "cat"]).alias("_x"))
h = d.filter(pl.col("rank") <= K).sort(["fidx", "rank"])
h = h.with_columns(pl.col("label").sum().over("fidx").alias("hp"), pl.len().over(["fidx", "cat"]).alias("cat_n_head")).filter(pl.col("hp") > 0)
h = h.with_columns(pl.col("cat_n_head").cast(pl.Float64).log().alias("lcatn_head"), pl.col("same_as_top").cast(pl.Float64))
off, sz = kit.feed_layout(h)
y = h["label"].to_numpy().astype(np.float64); s = h["score"].to_numpy().astype(np.float64)
a0, _, _, L0 = clogit(np.zeros((len(s), 0)), s, y, off, sz)
print("K", K, "feeds", len(sz), "pos", y.sum(), "scale", a0)
rows = []
for c in BASEF + ["lcatn_head", "same_as_top"] + [x + suf for x in BASEF for suf in ("_dm", "_pr")]:
    X = zcols(h, [c])
    if X.std() == 0:
        continue
    a, b, se, L = clogit(X, s, y, off, sz, l2=0)
    rows.append((c, b[0], b[0] / se[0], L0 - L))
rows.sort(key=lambda r: -abs(r[2]))
for r in rows[:30]: print("%-22s %+.4f z=%+.2f dLL=%.1f" % r)
