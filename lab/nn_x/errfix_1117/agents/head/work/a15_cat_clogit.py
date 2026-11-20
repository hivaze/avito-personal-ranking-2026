import hk, clogit, numpy as np, polars as pl, sys
c = hk.Ctx(); r = c.col("rank"); fid = c.col("fidx"); sc = c.score; cat = c.col("cat")
for K in [5, 10, 30, 300]:
    inh = r <= K
    P, S, Soff = clogit.build_sets(fid, c.y, inh)
    out = []
    for v in np.unique(cat):
        x = (cat == v).astype(float)
        if x[inh].sum() < 300: continue
        b, se, ll = clogit.fit(np.column_stack([sc, x]), P, S, Soff)
        out.append((v, int(x[inh].sum()), int((x[P] == 1).sum()), b[1], b[1] / se[1]))
    out.sort(key=lambda t: -abs(t[4]))
    print(f"K={K}: " + " | ".join(f"cat{t[0]} n={t[1]} pos={t[2]} b={t[3]:+.2f} z={t[4]:+.1f}" for t in out[:8]))
