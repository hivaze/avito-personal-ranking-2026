"""Fresher copy of a near-duplicate pair: deeper look + cross-fit."""
from lib import *
D = Data(); df = D.df
npos = np.add.reduceat(D.y, D.off)
p = softmax_p(D.s, D.off, D.sz, npos, 1.0)
cm = D.col("cand_maxcos"); f4 = D.col("f4"); xp = D.col("xp_imp_c50"); rank = D.col("rank"); s = D.s
fid = D.feed_of_row


def partner_diff(thr, x):
    """for rows in a mutual-nearest group (equal cand_maxcos >= thr within feed): x - mean(x of the other members); else 0.
    also returns group-size array"""
    m = cm >= thr
    t = pl.DataFrame({"f": fid, "c": np.where(m, cm, np.nan), "x": x, "i": np.arange(len(x))})
    g = t.filter(pl.col("c").is_not_nan()).with_columns(
        pl.len().over(["f", "c"]).alias("n"), pl.col("x").sum().over(["f", "c"]).alias("sx"))
    out = np.zeros(len(x)); n = np.zeros(len(x))
    i = g["i"].to_numpy(); nn = g["n"].to_numpy(); sx = g["sx"].to_numpy(); xx = g["x"].to_numpy()
    ok = nn >= 2
    out[i[ok]] = xx[ok] - (sx[ok] - xx[ok]) / (nn[ok] - 1)
    n[i] = nn
    return out, n


for thr in [0.95, 0.97, 0.99]:
    d4, n = partner_diff(thr, f4)
    inpair = n >= 2
    print(f"\nthr {thr}: rows in groups {inpair.sum()}, group sizes {np.bincount(n[inpair].astype(int))[2:6]}")
    for lo, hi in [(-1, -0.2), (-0.2, -0.05), (-0.05, -0.005), (-0.005, 0.005), (0.005, 0.05), (0.05, 0.2), (0.2, 1)]:
        mm = inpair & (d4 >= lo) & (d4 < hi)
        if mm.sum() < 200: continue
        a, e = D.y[mm].sum(), p[mm].sum()
        print("  f4 - partner in [%6.3f,%6.3f): n %6d pos %4d A/E %.3f z %+.1f  mean rank %.0f" % (lo, hi, mm.sum(), a, a / e, (a - e) / np.sqrt(e), rank[mm].mean()))
NV = 0
def run(name, g, ws):
    global NV
    NV += len(ws)
    r = D.crossfit(lambda w: s + w * g + 1e-7 * s, ws)
    best = max(r["insample"], key=lambda t: t[1])
    print(f"{name:38s} CF {r['cf']:+.5f} se {r['se']:.5f} z {r['z']:5.2f} | in-sample best w={best[0]} {best[1]:+.5f} ({best[2]:.5f}) | chosen {r['chosen']}")
    for g_, m_, s_ in r["insample"]:
        print("      w=%s %+.5f (%.5f)" % (g_, m_, s_))
for thr in [0.95, 0.97, 0.99]:
    d4, n = partner_diff(thr, f4)
    run(f"fresher-copy indicator thr {thr}", (d4 < -0.005).astype(float), [0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    run(f"-(f4 - partner) thr {thr}", -np.clip(d4, -0.5, 0.5), [0.1, 0.2, 0.4, 0.6, 1.0])
print("variants", NV)
