import hk, numpy as np, polars as pl
c = hk.Ctx(); df = c.df
a = c.col("s_ft").astype(float); b = c.col("s_to").astype(float)
ws = [0.0, 0.3, 0.5, 0.7, 1.0]
ND = {w: c.nd(w * a + (1 - w) * b) - c.base for w in ws}
wd = c.col("weekday")[c.off]; fx = c.col("fidx")[c.off]
print("fidx range by weekday:")
for v in np.unique(wd):
    m = wd == v
    print(f" wd={v} n={m.sum()} fidx med={np.median(fx[m]):.0f} base={c.base[m].mean():.4f} " + " ".join(f"w{w}:{ND[w][m].mean()*1e3:+.2f}" for w in ws))
q = np.digitize(fx, np.quantile(fx, [.2, .4, .6, .8]))
for v in range(5):
    m = q == v
    print(f" fidx quintile {v}: base={c.base[m].mean():.4f} " + " ".join(f"w{w}:{ND[w][m].mean()*1e3:+.2f}" for w in ws))
