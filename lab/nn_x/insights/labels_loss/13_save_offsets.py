"""Save the GPU-ready exploration examination offsets: g_noend[r] = log(CTR at shown within-feed rank r / base),
fit on the 80K proxy pool feeds (random-like, M>=50, holdout excluded) using non-end slots only (rank-from-end >= 3),
smoothed beyond rank 30 (+-10 running mean), r = 0..299; also indexed by raw pos (gaps are ~2% in random feeds).
Adds g_noend to out/pool_pos_offset.npz (keeps g_rank / h_rkend of 05)."""
import numpy as np
D = "out/proxy"; o5 = dict(np.load("out/pool_pos_offset.npz"))
y = np.load(f"{D}/pool_y.npy").astype(np.float64); aux = np.load(f"{D}/pool_aux.npy"); rk, rke = aux[0], aux[1]
m = rke >= 3; base = y[m].mean(); g = np.zeros(300)
for r in range(300):
    mm = m & (np.minimum(rk, 120) == min(r, 120)); g[r] = np.log(max(y[mm].sum(), 1) / max(mm.sum() * base, 1e-9))
gs = g.copy()
for r in range(30, 300): gs[r] = g[max(30, r - 10):min(299, r + 10) + 1].mean()
o5["g_noend"] = gs; np.savez("out/pool_pos_offset.npz", **o5)
print("g_noend at 0,1,2,3,5,10,20,30,50,100,200:", " ".join(f"{gs[r]:+.3f}" for r in [0,1,2,3,5,10,20,30,50,100,200]))
print("rows used", m.sum(), "base", base)
