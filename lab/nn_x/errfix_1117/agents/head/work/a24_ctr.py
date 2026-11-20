import hk, clogit, numpy as np
c = hk.Ctx(); r = c.col("rank"); fid = c.col("fidx"); sc = c.score; y = c.y
f5 = c.col("f5").astype(float); xp = c.col("xp_imp_c50").astype(float)
n = np.expm1(xp)
print("f5 by xp bucket:")
for lo, hi in [(0, .01), (.01, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 3.92), (3.92, 9)]:
    m = (xp >= lo) & (xp < hi)
    print(f"  xp[{lo},{hi}) n={m.sum()} f5 mean={f5[m].mean():.3f} sd={f5[m].std():.3f} label={y[m].mean():.4f} q={np.quantile(f5[m],[.1,.5,.9]).round(2)}")
prior = np.average(f5[xp > 3.9])
for K in [10, 30, 300]:
    inh = r <= K
    P, S, Soff = clogit.build_sets(fid, y, inh)
    for a in [2, 5, 10, 20]:
        sm = (f5 * n + a * prior) / (n + a)
        z = (sm - sm[inh].mean()) / sm[inh].std()
        b, se, ll = clogit.fit(np.column_stack([sc, z]), P, S, Soff)
        print(f"K={K} smoothed ctr a={a}: b={b[1]:+.3f} z={b[1]/se[1]:+.2f}")
    # f5 among well-exposed only
    z = np.where(xp > 3.9, (f5 - f5[inh].mean()) / f5[inh].std(), 0)
    b, se, ll = clogit.fit(np.column_stack([sc, z, (xp > 3.9).astype(float)]), P, S, Soff)
    print(f"K={K} f5 x exposed: b={b[1]:+.3f} z={b[1]/se[1]:+.2f}; exposed main {b[2]:+.3f} z={b[2]/se[2]:+.2f}")
