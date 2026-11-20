"""Feed flatness (score spread in the head) x feature slopes, in the head-restricted clogit (top-30)."""
from wc import *
df = load(); d = features(df)
d = d.with_columns((pl.col("score").filter(pl.col("rank") == 1).first().over("fidx") - pl.col("score").filter(pl.col("rank") == pl.col("M").clip(None, 20)).first().over("fidx")).alias("spread20"))
h = d.filter(pl.col("rank") <= 30).sort(["fidx", "rank"])
h = h.with_columns(pl.col("label").sum().over("fidx").alias("hp")).filter(pl.col("hp") > 0)
off, sz = kit.feed_layout(h)
y = h["label"].to_numpy().astype(float); s = h["score"].to_numpy().astype(float)
sp = h["spread20"].to_numpy()
q = np.quantile(sp[off], [1 / 3, 2 / 3]); print("spread terciles", q)
flat = sp <= q[0]; steep = sp > q[1]
F = ["pre2_fe_z", "pre2_new_z", "ucos", "f1", "hmax", "h5", "h1", "lcv", "xp_imp_c50", "lpop", "f4", "f5", "ldup", "sdiff", "f3", "lcatn", "cat_hist_share", "f2"]
X0 = zcols(h, F)
rows = []
for nm, m in [("flat", flat), ("steep", steep)]:
    for k, c in enumerate(F):
        X = (X0[:, k] * m)[:, None]
        a, b, se, L = clogit(X, s, y, off, sz, l2=0)
        rows.append((nm, c, b[0], b[0] / se[0]))
    # score scale in segment
    a, b, se, L = clogit((s * m)[:, None], s, y, off, sz, l2=0)
    rows.append((nm, "score*seg", b[0], b[0] / se[0]))
rows.sort(key=lambda r: -abs(r[3]))
for r in rows[:20]: print("%-6s %-15s %+.4f z=%+.2f" % r)
