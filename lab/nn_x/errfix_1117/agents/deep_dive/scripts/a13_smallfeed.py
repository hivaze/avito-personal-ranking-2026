"""Follow-up: similarity boost in small feeds. Check M dependence, alternative similarity features, fold seeds."""
from util import *

df = load()
c = Ctx(df)
s = c.base


def col(n):
    return df[n].to_numpy().astype(np.float64)


def nz(a, v):
    a = a.copy(); a[~np.isfinite(a)] = v; return a


def zf(a):
    a = nz(a, np.nanmean(a))
    m = np.bincount(c.fid, weights=a) / c.sz
    v = np.bincount(c.fid, weights=a * a) / c.sz - m * m
    return (a - m[c.fid]) / np.sqrt(np.maximum(v[c.fid], 1e-6))


M = col("M")
W = [0.02, 0.05, 0.08, 0.1, 0.15]
hz = zf(nz(col("hmax"), 0.5))
print("M-bin deltas for w=0.05 (in-sample, per bin):")
d = c.feed_delta(s + 0.05 * hz)
Mf = M[c.off]
for lo, hi in [(5, 15), (15, 25), (25, 40), (40, 60), (60, 100), (100, 200), (200, 300)]:
    m = (Mf >= lo) & (Mf < hi)
    print(f"  M[{lo},{hi}) feeds {m.sum()} delta {d[m].mean():+.5f} se {d[m].std(ddof=1) / np.sqrt(m.sum()):.5f}")
for feat in ["hmax", "h32", "h10", "h5", "hmean5", "c_vfeed", "c_vsearch", "c_contact", "ucos", "pre2_fe_z", "pre2_new_z"]:
    g = zf(nz(col(feat), np.nanmedian(col(feat))))
    for thr in [40]:
        x = (M < thr) * g
        r = c.crossfit(lambda w: s + w * x, [-0.1, -0.05] + W)
        print(f"{feat:10s} M<{thr}: cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins={r['ins_delta']:+.5f}")
# alternative fold seeds for the hmax rule
for seed in [1, 2, 3]:
    c.folds_row = kit.folds(df, seed=seed); c.folds_feed = c.folds_row[c.off]
    x = (M < 40) * hz
    r = c.crossfit(lambda w: s + w * x, [-0.1, -0.05, -0.02] + W)
    print(f"seed {seed}: cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']}")
