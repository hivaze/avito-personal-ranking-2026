"""Substitution/cascade within category: pen_i = sum of model probabilities of higher-scored same-category items.
p = npos_hat * softmax(a*score) within feed. new = score - w * pen (or log(1 - rho*pen))."""
import fx, kit, numpy as np, polars as pl
df = kit.load(); c = fx.Ctx(df); s = c.base; nd = c.base_nd
feed = c.feed
cat = df["cat"].to_numpy().astype(np.int64)
M = df["M"].to_numpy().astype(float); Mf = M[c.off]
cr = df["cat_rank_in_feed"].to_numpy().astype(float)

def softmax_feed(z):
    zmax = np.full(c.nf, -np.inf); np.maximum.at(zmax, feed, z)
    e = np.exp(z - zmax[feed]); se = np.bincount(feed, weights=e, minlength=c.nf)
    return e / se[feed]

def cum_above_same_cat(p):
    """sum of p over same-feed same-cat items with higher score."""
    order = np.lexsort((-s, cat, feed))          # by feed, cat, score desc
    pp = p[order]
    key = feed[order] * 64 + cat[order]
    cs = np.cumsum(pp)
    start = np.r_[0, np.nonzero(key[1:] != key[:-1])[0] + 1]
    grp_start = np.repeat(start, np.diff(np.r_[start, len(key)]))
    base = np.where(grp_start > 0, cs[grp_start - 1], 0.0)
    excl = cs - pp - base                         # sum of predecessors in the group
    out = np.empty_like(p); out[order] = excl
    return out

nvar = 0
def run(nm, x, grid):
    global nvar
    nvar += len(grid) - 1
    mean, se, z, ch, D = c.crossfit(lambda w: s - w * x, list(grid))
    segtxt = " | ".join(f"{lab}:" + ",".join(f"{d*1e4:+.1f}" for d in D[:, m].mean(1)[1:])
                        for lab, m in [("M<30", Mf < 30), ("30-100", (Mf >= 30) & (Mf < 100)), ("100+", Mf >= 100)])
    print(f"{nm:28s} ins " + " ".join(f"{d*1e4:+5.1f}" for d in D.mean(1)[1:]) + f" CF {mean*1e4:+.2f} se {se*1e4:.2f} z {z:+.2f} ch {ch}")
    print("        ", segtxt, flush=True)

for a in (0.5, 1.0, 2.0):
    p = softmax_feed(a * s)
    pen = cum_above_same_cat(p)
    print(f"a={a}: pen quantiles", np.quantile(pen, [.5, .9, .99]).round(4), "max", pen.max().round(3))
    run(f"pen a={a}", pen, (0.0, 0.25, 0.5, 1.0, 2.0, 4.0))
    run(f"sqrt pen a={a}", np.sqrt(pen), (0.0, 0.02, 0.05, 0.1, 0.2, 0.4))
# count-based with finer grid near 0 for comparison
run("(cr-1)^0.3 fine", (cr - 1) ** 0.3, (0.0, 0.005, 0.01, 0.015, 0.02, 0.03))
run("[cr>=2]", (cr >= 2).astype(float), (0.0, 0.005, 0.01, 0.02, 0.03, 0.05))
print("variants in this script", nvar)
