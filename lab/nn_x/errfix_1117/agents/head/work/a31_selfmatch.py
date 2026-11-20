import hk, numpy as np
c = hk.Ctx(); y = c.y; r = c.col("rank")
Mb = np.digitize(c.col("M"), [30, 60, 100, 200]); key = Mb * 1000 + np.minimum(r, 300)
uk, inv = np.unique(key, return_inverse=True); rate = (np.bincount(inv, weights=y) / np.bincount(inv))[inv]
for nm in ["c_fav", "c_contact", "c_order", "c_lead", "c_vfeed", "c_vsearch", "hmax", "h1"]:
    x = c.col(nm).astype(float)
    print(f"== {nm}")
    for lo, hi in [(0.95, 0.98), (0.98, 0.99), (0.99, 0.995), (0.995, 1.01)]:
        for rlo, rhi in [(1, 10), (11, 300)]:
            m = (x >= lo) & (x < hi) & (r >= rlo) & (r <= rhi)
            if m.sum() < 30: continue
            print(f"   [{lo},{hi}) ranks {rlo}-{rhi}: n={m.sum():6d} pos={int(y[m].sum()):4d} rate={y[m].mean():.4f} exp={rate[m].mean():.4f} ratio={y[m].mean()/rate[m].mean():.2f}")
