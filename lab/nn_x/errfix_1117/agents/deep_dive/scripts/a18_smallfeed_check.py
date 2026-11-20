"""Robustness of the small-feed hmax rule: per-fold OOF deltas, thresholds, weights, rank of effect, by-segment."""
from util import *

df = load()
c = Ctx(df)
s = c.base


def zf(a):
    a = a.copy(); a[~np.isfinite(a)] = np.nanmean(a)
    m = np.bincount(c.fid, weights=a) / c.sz
    v = np.bincount(c.fid, weights=a * a) / c.sz - m * m
    return (a - m[c.fid]) / np.sqrt(np.maximum(v[c.fid], 1e-6))


hm = df["hmax"].to_numpy().astype(np.float64)
hz = zf(hm)
M = df["M"].to_numpy()
for thr in [20, 30, 40, 50, 60]:
    x = (M < thr) * hz
    r = c.crossfit(lambda w: s + w * x, [0.02, 0.035, 0.05, 0.075, 0.1])
    per = [r["oof"][c.folds_feed == f].mean() for f in range(5)]
    print(f"M<{thr}: cf={r['cf']:+.5f} z={r['z']:+.2f} chosen={r['chosen']} per-fold={np.round(per, 5)}")
for w in [0.02, 0.035, 0.05, 0.075, 0.1, 0.15]:
    d = c.feed_delta(s + w * (M < 40) * hz)
    print(f"  w={w}: in-sample {d.mean():+.5f} se {d.std(ddof=1) / np.sqrt(len(d)):.5f} changed feeds {(d != 0).mean():.3f}")
# raw hmax instead of within-feed z
for w in [0.1, 0.2, 0.3, 0.5]:
    hraw = np.where(np.isfinite(hm), hm, 0.5)
    d = c.feed_delta(s + w * (M < 40) * hraw)
    print(f"  raw hmax w={w}: in-sample {d.mean():+.5f}")
# where does it act: guest / newest_type / H
d = c.feed_delta(s + 0.05 * (M < 40) * hz)
fl = df.group_by("fidx", maintain_order=True).agg(pl.col("M").first(), pl.col("guest").first(), pl.col("newest_type").first(),
                                                  pl.col("n_hist_cats32").first(), pl.col("H").first())
small = fl["M"].to_numpy() < 40
for name, key in [("guest", fl["guest"].to_numpy()), ("H<=50", fl["H"].to_numpy() <= 50), ("hcats<=2", fl["n_hist_cats32"].to_numpy() <= 2),
                  ("newest view", fl["newest_type"].to_numpy() == 1)]:
    for v in [False, True]:
        m = small & (key == v)
        print(f"  small & {name}={v}: feeds {m.sum()} delta {d[m].mean():+.5f} se {d[m].std(ddof=1) / np.sqrt(max(m.sum(), 2)):.5f}")
