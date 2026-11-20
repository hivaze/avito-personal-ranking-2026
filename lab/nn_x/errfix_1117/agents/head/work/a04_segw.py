"""Branch weight w (on s_ft) by feed segment, full list. Per-segment delta curves."""
import hk, numpy as np, polars as pl

c = hk.Ctx(); df = c.df
a = c.col("s_ft").astype(float); b = c.col("s_to").astype(float)
ws = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
ND = {w: c.nd(w * a + (1 - w) * b) - c.base for w in ws}
fe = df.group_by("fidx", maintain_order=True).agg(pl.all().first())
segs = {
    "M": (fe["M"].to_numpy(), [0, 30, 60, 100, 200, 300]),
    "H": (fe["H"].to_numpy(), [-1, 10, 50, 100, 150, 200, 255, 256]),
    "guest": (fe["guest"].to_numpy().astype(int), None),
    "uvec_missing": (fe["uvec_missing"].to_numpy().astype(int), None),
    "newest_type": (fe["newest_type"].to_numpy(), None),
    "newest_src": (fe["newest_src"].to_numpy(), None),
    "n_hist_cats32": (fe["n_hist_cats32"].to_numpy(), [-1, 1, 2, 4, 8, 32]),
    "n_feed_cats": (fe["n_feed_cats"].to_numpy(), [0, 1, 3, 6, 10, 60]),
    "hour": (fe["hour"].to_numpy(), [-1, 6, 12, 18, 24]),
}
for name, (v, cuts) in segs.items():
    print("==", name)
    if cuts is None:
        groups = [(str(u), v == u) for u in np.unique(v)]
    else:
        groups = [(f"({lo},{hi}]", (v > lo) & (v <= hi)) for lo, hi in zip(cuts[:-1], cuts[1:])]
    for g, m in groups:
        if m.sum() < 80:
            continue
        line = " ".join(f"{w}:{ND[w][m].mean()*1e3:+.2f}" for w in ws)
        print(f"  {g:12s} n={m.sum():5d} {line}  (x1e-3)")
