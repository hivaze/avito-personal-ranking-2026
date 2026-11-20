"""A7: when the FT and TO branches of G3 disagree, who is right? Per-feed FT-TO disagreement (1 - within-feed Pearson of
the branch z) deciles within M bins vs the pool-level FT gain; same on test for the distribution (public / private)."""
from lib import *  # noqa

vm, tm = meta("val"), meta("test")
Sv, St = segments("val"), segments("test")
rows = []
for name, ms in [("G3", POOLS["G3"]), ("F6", FAM["F6"]), ("F7x", FAM["F7x"])]:
    zt = zfeed(branch(ms, "val", "to"), "val"); zf = zfeed(branch(ms, "val", "ft"), "val")
    dis = 1 - np.bincount(vm["fid"], zt * zf, vm["n"]) / vm["M"]
    a, b, c = pf(zt), pf(zf), pf(0.5 * zt + 0.5 * zf)
    q = np.zeros(vm["n"], np.int64)
    for l in np.unique(Sv["M"]):
        i = np.where(Sv["M"] == l)[0]
        q[i] = (10 * np.argsort(np.argsort(dis[i]))) // len(i)
    print(f"{name}: FT-TO disagreement deciles (within M bins): dec | dis | FT-only - TO | final - TO | FT wins share (strict)")
    for d in range(10):
        i = q == d
        r = dict(pool=name, dec=d, dis=dis[i].mean(), ftonly=(b - a)[i].mean(), ftonly_se=pse((b - a)[i])[1],
                 fin=(c - a)[i].mean(), fin_se=pse((c - a)[i])[1], ftwin=((b > a)[i].mean()), towin=((a > b)[i].mean()))
        rows.append(r)
        print(f"  {d} | {r['dis']:.4f} | {r['ftonly']:+.5f}+-{r['ftonly_se']:.5f} | {r['fin']:+.5f}+-{r['fin_se']:.5f} | "
              f"FT>TO {r['ftwin']:.3f} TO>FT {r['towin']:.3f}")
    ztt = zfeed(branch(ms, "test", "to"), "test"); zft = zfeed(branch(ms, "test", "ft"), "test")
    dist = 1 - np.bincount(tm["fid"], ztt * zft, tm["n"]) / tm["M"]
    pub = St["day"] == "public"
    print(f"  mean FT-TO disagreement val {dis.mean():.4f} | test {dist.mean():.4f} (public {dist[pub].mean():.4f} private {dist[~pub].mean():.4f})")
pl.DataFrame(rows).write_csv(OUT / "a7_ft_disagree.csv")
