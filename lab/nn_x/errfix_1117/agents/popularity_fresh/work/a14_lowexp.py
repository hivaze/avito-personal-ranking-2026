"""Low-but-nonzero exposure items at the head: A/E by f5 and by exact impression count; cross-fit a demotion."""
from lib import *
D = Data()
npos = np.add.reduceat(D.y, D.off)
p = softmax_p(D.s, D.off, D.sz, npos, 1.0)
xp = D.col("xp_imp_c50"); f5 = D.col("f5"); f4 = D.col("f4"); rank = D.col("rank")
imp = np.round(np.expm1(xp)).astype(int)
for rn, rm in [("rank1", rank == 1), ("rank1-3", rank <= 3), ("rank1-10", rank <= 10), ("rank11-50", (rank > 10) & (rank <= 50))]:
    print(f"\n== {rn}")
    for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 20), (21, 49), (50, 50)]:
        m = rm & (imp >= lo) & (imp <= hi)
        a, e = D.y[m].sum(), p[m].sum()
        line = "  imp %2d-%2d n %6d A/E %.3f z %5.2f" % (lo, hi, m.sum(), a / e, (a - e) / np.sqrt(e))
        for fl, fh in [(0, .4), (.4, .8), (.8, 1)]:
            mm = m & (f5 >= fl) & (f5 < fh)
            a, e = D.y[mm].sum(), p[mm].sum()
            line += " | f5[%.1f,%.1f) %.2f(%4.1f)" % (fl, fh, a / e, (a - e) / np.sqrt(e))
        print(line)
s = D.s
for hd in [1, 3, 10]:
    for lo, hi in [(1, 2), (1, 4), (1, 9)]:
        g = ((imp >= lo) & (imp <= hi) & (rank <= hd)).astype(float)
        r = D.crossfit(lambda w: s + w * g + 1e-7 * s, [-0.02, -0.05, -0.1, -0.2, -0.3])
        best = max(r["insample"], key=lambda t: t[1])
        print("demote imp %d-%d in top %2d: CF %+.5f se %.5f z %.2f | best in-sample w=%s %+.5f" % (lo, hi, hd, r["cf"], r["se"], r["z"], best[0], best[1]))
