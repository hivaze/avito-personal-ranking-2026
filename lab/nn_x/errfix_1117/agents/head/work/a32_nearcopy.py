import hk, numpy as np
c = hk.Ctx(); s = c.score; r = c.col("rank"); y = c.y
hm = np.nan_to_num(c.col("hmax").astype(float), nan=0); hidx = c.col("hmax_idx")
cc = np.nan_to_num(c.col("c_contact").astype(float), nan=0)
Mb = np.digitize(c.col("M"), [30, 60, 100, 200]); key = Mb * 1000 + np.minimum(r, 300)
uk, inv = np.unique(key, return_inverse=True); rate = (np.bincount(inv, weights=y) / np.bincount(inv))[inv]
m = (hm >= .995) & (r <= 10)
print("hmax>=.995 in head, rank distribution:", np.bincount(r[m], minlength=11)[1:])
for lo, hi in [(0, 4), (5, 24), (25, 99), (100, 300)]:
    mm = m & (hidx >= lo) & (hidx <= hi)
    print(f"  hidx {lo}-{hi}: n={mm.sum()} rate={y[mm].mean():.4f} exp={rate[mm].mean():.4f}")
grid = [(T, K, w) for T in [0.99, 0.995] for K in [10, 20] for w in [0.0, 0.05, 0.1, 0.2, 0.3]]
mcf, se, ch, oof, D = c.crossfit(lambda p: s + p[2] * ((hm >= p[0]) & (r <= p[1])), grid)
print(f"hmax near-copy head boost: cf={mcf:+.5f} se={se:.5f} z={mcf/se if se else 0:+.2f} chosen={ch}")
print("   ", " ".join(f"{p}:{D[p].mean()*1e4:+.1f}" for p in grid if p[2] > 0))
grid = [(T, K, w) for T in [0.99, 0.995] for K in [10, 20] for w in [0.0, 0.05, 0.1, 0.2, 0.3]]
mcf, se, ch, oof, D = c.crossfit(lambda p: s + p[2] * ((cc >= p[0]) & (r <= p[1])), grid)
print(f"c_contact near-copy head boost: cf={mcf:+.5f} se={se:.5f} z={mcf/se if se else 0:+.2f} chosen={ch}")
print("   ", " ".join(f"{p}:{D[p].mean()*1e4:+.1f}" for p in grid if p[2] > 0))
