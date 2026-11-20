import hk, clogit, numpy as np
c = hk.Ctx(); r = c.col("rank"); fid = c.col("fidx"); sc = c.score; y = c.y
x = c.col("it_hist_et1_luo").astype(float); xa = c.col("it_hist_all_luo").astype(float)
print("quantiles et1 all rows", np.quantile(x, [.5, .9, .99, .995, .999]).round(0), "max", x.max())
Mb = np.digitize(c.col("M"), [30, 60, 100, 200]); key = Mb * 1000 + np.minimum(r, 300)
uk, inv = np.unique(key, return_inverse=True); rate = (np.bincount(inv, weights=y) / np.bincount(inv))[inv]
for lo, hi in [(0, 50), (50, 150), (150, 300), (300, 500), (500, 800), (800, 1500), (1500, 1e9)]:
    m = (x >= lo) & (x < hi)
    print(f"et1 [{lo},{hi}) n={m.sum():6d} pos={y[m].sum():5d} rate={y[m].mean():.4f} exp={rate[m].mean():.4f} ratio={y[m].mean()/rate[m].mean():.2f} med_rank={np.median(r[m]):.0f}")
inh = np.ones(len(y), bool)
P, S, Soff = clogit.build_sets(fid, y, inh)
for thr in [200, 300, 500, 800]:
    b, se, ll = clogit.fit(np.column_stack([sc, (x >= thr).astype(float)]), P, S, Soff)
    print(f"full-list clogit et1>={thr}: b={b[1]:+.3f} z={b[1]/se[1]:+.2f} n={int((x>=thr).sum())}")
lx = np.log1p(x)
b, se, ll = clogit.fit(np.column_stack([sc, (lx - lx.mean()) / lx.std(), np.maximum(lx - np.log1p(300), 0)]), P, S, Soff)
print("full: log + hinge@300", b, b / se)
