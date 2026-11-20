"""D3 (DECLARATION AMENDMENT 4): non-linear interaction of 'shown elsewhere' with the pool score, direct NDCG, nested."""
import json, time
import numpy as np
from rlib import OUT, Val, ndcg_feeds, ndcg_grid

t0 = time.time()
V = Val()
C = np.load(f"{OUT}/my_counts.npy")[:V.N].astype(np.float64)
GRID = np.array(sorted([0.0] + [s * v for s in (1, -1) for v in (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)]))
G0 = int(np.flatnonzero(GRID == 0.0)[0])
folds = [V.folds(s) for s in (11, 12, 13)]
cold = V.cold.astype(np.float64); b = V.b
NAMES = ["i_a*[b>0]", "i_a*[b>1]", "i_a*[b>2]", "i_a*max(b,0)", "cs_a*[b>0]", "cs_a*[b>1]", "[n_a>=2]*[b>0]", "[n_a>=2]*[b>1]"]


def cols_of(Cx):
    n = Cx[:, 0]; i = (n >= 1).astype(np.float64); i2 = (n >= 2).astype(np.float64)
    return [i * (b > 0), i * (b > 1), i * (b > 2), i * np.maximum(b, 0), cold * i * (b > 0), cold * i * (b > 1), i2 * (b > 0), i2 * (b > 1)]


def read(Cx):
    cols = cols_of(Cx); K = len(cols)
    P = np.empty((K, len(GRID), V.nf))
    for j in range(K):
        ndcg_grid(b, np.ascontiguousarray(cols[j].astype(np.float64)), GRID, V.y, V.off, V.M, P[j])
    d1 = []; d2 = []; buf = np.empty((len(GRID), V.nf)); picks = []
    for fold in folds:
        o1 = np.empty(V.nf); o2 = np.empty(V.nf)
        for k in range(5):
            tr = fold != k
            mt = P[:, :, tr].mean(2); j, g = np.unravel_index(np.argmax(mt), mt.shape)
            if mt[j, g] <= mt[0, G0] + 1e-7:
                j, g = 0, G0
            o1[~tr] = P[j, g][~tr]; picks.append((NAMES[j], float(GRID[g])))
            w = np.zeros(K); base = b.copy(); cur = V.base[tr].mean()
            for _ in range(2):
                for jj in range(K):
                    bm = base - w[jj] * cols[jj]
                    ndcg_grid(bm, np.ascontiguousarray(cols[jj].astype(np.float64)), GRID, V.y, V.off, V.M, buf)
                    m2 = buf[:, tr].mean(1); gg = int(np.argmax(m2))
                    if m2[gg] > cur + 1e-7:
                        w[jj] = GRID[gg]; base = bm + w[jj] * cols[jj]; cur = m2[gg]
            o2[~tr] = ndcg_feeds(base, V.y, V.off, V.M)[~tr]
        d1.append(o1 - V.base); d2.append(o2 - V.base)
    mt = P.mean(2) - V.base.mean(); j, g = np.unravel_index(np.argmax(mt), mt.shape)
    return np.mean(d1, 0), np.mean(d2, 0), (NAMES[j], float(GRID[g]), float(mt[j, g]), float((P[j, g] - V.base).std() / np.sqrt(V.nf))), picks, {NAMES[jj]: [float(GRID[int(np.argmax(mt[jj]))]), float(mt[jj].max())] for jj in range(K)}


st = lambda d: [float(d.mean()), float(d.std() / np.sqrt(len(d))), float(d.mean() / (d.std() / np.sqrt(len(d)))) if d.std() > 0 else 0.0]
d1, d2, ceil, picks, per = read(C)
res = dict(D3_1=st(d1), D3_2=st(d2), ceiling=ceil, picks=picks, per_column_insample=per, placebo=[])
print(f"REAL D3-1 {st(d1)[0]:+.5f} +- {st(d1)[1]:.5f} (z {st(d1)[2]:+.2f}) | D3-2 {st(d2)[0]:+.5f} +- {st(d2)[1]:.5f} (z {st(d2)[2]:+.2f})")
print(f"in-sample ceiling D3-1: {ceil[0]} w {ceil[1]:+.3f} gain {ceil[2]:+.5f} +- {ceil[3]:.5f}; per column: " + ", ".join(f"{k} {v[1]:+.5f}@{v[0]:+.3f}" for k, v in per.items()))
print("picks:", picks, flush=True)
grp = V.fid * 2 + V.cold.astype(np.int64)
for seed in range(500, 510):
    u = np.random.default_rng(seed).random(V.N)
    so = np.argsort(grp, kind="stable"); sr = np.lexsort((u, grp)); src = np.empty(V.N, np.int64); src[so] = sr
    p1, p2, pc, _, _ = read(C[src])
    res["placebo"].append([st(p1)[0], st(p2)[0], pc[2]])
    print(f"[WF] seed {seed}: D3-1 {st(p1)[0]:+.5f} D3-2 {st(p2)[0]:+.5f} | in-sample ceiling {pc[2]:+.5f} | {time.time() - t0:.0f}s", flush=True)
pp = np.array(res["placebo"])
for i, nm in enumerate(("D3_1", "D3_2")):
    print(f"placebo {nm}: mean {pp[:, i].mean():+.5f} sd {pp[:, i].std(ddof=1):.5f} min {pp[:, i].min():+.5f} max {pp[:, i].max():+.5f} | real {res[nm][0]:+.5f} above {(res[nm][0] > pp[:, i]).sum()} of {len(pp)}")
print(f"placebo in-sample ceilings: mean {pp[:, 2].mean():+.5f} max {pp[:, 2].max():+.5f} | real {ceil[2]:+.5f} above {(ceil[2] > pp[:, 2]).sum()} of {len(pp)}")
json.dump(res, open(f"{OUT}/r10_d3.json", "w"), indent=1)
print(f"DONE {time.time() - t0:.0f}s")
