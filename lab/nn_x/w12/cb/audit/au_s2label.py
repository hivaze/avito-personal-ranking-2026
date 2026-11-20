"""audit 3g: is the part of the residual that a new block explains label-relevant? Light: 2 threads."""
from au_common import *
light(2)
m = Meta()
for k, sp in (("V", "val"), ("H", "hold")):
    if not (OUT / f"s1_{k}.npy").exists(): continue
    y, off, M = m.y[sp], m.off[sp], m.M[sp]; nF = len(M); fid = np.repeat(np.arange(nF), M); P = y > 0
    Z = w16_pool(m, sp); s1 = np.load(OUT / f"s1_{k}.npy"); r = Z - s1
    cen = lambda x: x - (np.bincount(fid, x, nF) / M)[fid]
    wc = lambda a, b: float((cen(a) * cen(b)).sum() / np.sqrt((cen(a) ** 2).sum() * (cen(b) ** 2).sum()))
    print(f"\n== {sp}: mean on positives: z(W16) {Z[P].mean():.4f} = tree fit {cen(s1.astype(np.float64))[P].mean():.4f} + residual {cen(r.astype(np.float64))[P].mean():.4f}; within-feed corr with label: W16 {wc(Z, y):.4f} fit {wc(s1, y):.4f} residual {wc(r, y):.4f}")
    for b in ("same177", "seq", "trd", "emb", "prod", "all_new"):
        f = OUT / f"s2_{k}_{b}.npy"
        if not f.exists(): continue
        p2 = np.load(f).astype(np.float64); left = r - p2
        bj, sj, lj = clogit(np.stack([s1, p2], 1), y.astype(np.float64), off, M); b1, s1e, l1 = clogit(s1[:, None], y.astype(np.float64), off, M)
        print(f"   {b:8s}: explains {100*(1 - np.mean(cen(left)**2)/np.mean(cen(r.astype(np.float64))**2)):5.1f}% of residual variance; mean on positives: explained part {cen(p2)[P].mean():+.4f} +- {cen(p2)[P].std()/np.sqrt(P.sum()):.4f}, "
              f"left {cen(left)[P].mean():+.4f}; corr(explained part, label) {wc(p2, y):+.4f}; clogit [fit, explained]: beta {bj[0]:.3f}, {bj[1]:+.3f} +- {sj[1]:.3f} (z {bj[1]/sj[1]:+.1f}), loss gain {(l1 - lj).mean():+.6f} nats/feed")
