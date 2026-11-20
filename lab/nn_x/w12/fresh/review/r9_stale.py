"""STALE-VAL read (DECLARATION AMENDMENT 3): D1 / D2 with the stale train-only pool as the base."""
import json, sys, time
import numpy as np
from rlib import OUT, ROOT, Val, ndcg_feeds, ndcg_grid, zfeed

t0 = time.time()
V = Val()
C = np.load(f"{OUT}/my_counts.npy")[:V.N].astype(np.float64)
GRID = np.array(sorted([0.0] + [s * v for s in (1, -1) for v in (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)]))
G0 = int(np.flatnonzero(GRID == 0.0)[0])
folds = [V.folds(s) for s in (11, 12, 13)]
cold = V.cold.astype(np.float64)
MEM = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in range(951, 959)]
z = lambda x: zfeed(np.asarray(x, np.float64), V.fid, V.M)
S = z(np.mean([z(np.load(f"{ROOT}/nn_x/w5/preds/stale/nn_{m}_staleval.npy")) for m in MEM], 0))
F = z(np.mean([z(np.load(f"{ROOT}/nn_x/w5/preds/nn_{m}_trainonly_val.npy")) for m in MEM], 0))
nS = ndcg_feeds(S, V.y, V.off, V.M); nF = ndcg_feeds(F, V.y, V.off, V.M)
st = lambda d: [float(d.mean()), float(d.std() / np.sqrt(len(d))), float(d.mean() / (d.std() / np.sqrt(len(d)))) if d.std() > 0 else 0.0]
print(f"16 members; fresh train-only pool F {nF.mean():.5f}, stale pool S {nS.mean():.5f}; staleness loss {st(nS - nF)[0]:+.5f} +- {st(nS - nF)[1]:.5f}; final file pool {V.base.mean():.5f}", flush=True)
h0 = np.load(f"{ROOT}/data/feats/val/it_hist_all_luo.npy"); h1 = np.load(f"{ROOT}/nn_x/r6/stale_cf/both_it_hist_all_luo.npy")
chg = (h0 != h1); madecold = (h0 > 0) & (h1 == 0)
print(f"scenario: it_hist_all_luo changed on {100 * chg.mean():.1f}% of val rows; set to 0 from > 0 on {100 * madecold.mean():.1f}%; "
      f"as-of coverage (n_a >= 1) among the rows made cold {100 * (C[madecold, 0] >= 1).mean():.1f}%, among the others {100 * (C[~madecold, 0] >= 1).mean():.1f}%")


def cols_A(Cx, b):
    n1, n2 = Cx[:, 0], Cx[:, 1]
    r1, z1 = V.forms(n1); _, z2 = V.forms(n2)
    i1 = (n1 >= 1).astype(np.float64); i2 = (n2 >= 1).astype(np.float64)
    return [z1, r1 - 0.5, i1, cold * i1, np.log1p(n1), z2, i2, b * i1, b * cold * i1]


def read(Cx, b, nb):
    cols = cols_A(Cx, b)
    P = np.empty((7, len(GRID), V.nf))
    for j in range(7):
        ndcg_grid(b, np.ascontiguousarray(cols[j]), GRID, V.y, V.off, V.M, P[j])
    d1 = []; d2 = []; buf = np.empty((len(GRID), V.nf))
    for fold in folds:
        o1 = np.empty(V.nf); o2 = np.empty(V.nf)
        for k in range(5):
            tr = fold != k
            mt = P[:, :, tr].mean(2); j, g = np.unravel_index(np.argmax(mt), mt.shape)
            if mt[j, g] <= mt[0, G0] + 1e-7:
                j, g = 0, G0
            o1[~tr] = P[j, g][~tr]
            w = np.zeros(9); base = b.copy(); cur = nb[tr].mean()
            for _ in range(2):
                for jj in range(9):
                    bm = base - w[jj] * cols[jj]
                    ndcg_grid(bm, np.ascontiguousarray(cols[jj]), GRID, V.y, V.off, V.M, buf)
                    m2 = buf[:, tr].mean(1); gg = int(np.argmax(m2))
                    if m2[gg] > cur + 1e-7:
                        w[jj] = GRID[gg]; base = bm + w[jj] * cols[jj]; cur = m2[gg]
            o2[~tr] = ndcg_feeds(base, V.y, V.off, V.M)[~tr]
        d1.append(o1 - nb); d2.append(o2 - nb)
    ceil = float((P.mean(2) - nb.mean()).max())
    return np.mean(d1, 0), np.mean(d2, 0), ceil


sys.path.insert(0, f"{ROOT}/src")
import pmix
W = pmix.weights()
ws = lambda d, w: [float(np.average(d, weights=w)), float(np.sqrt(np.average((d - np.average(d, weights=w)) ** 2, weights=w) / (w.sum() ** 2 / (w ** 2).sum())))]
res = {"ndcg_F": float(nF.mean()), "ndcg_S": float(nS.mean()), "loss": st(nS - nF)}
d1S, d2S, cS = read(C, S, nS); d1F, d2F, cF = read(C, F, nF)
res["real"] = dict(S_D1=st(d1S), S_D2=st(d2S), F_D1=st(d1F), F_D2=st(d2F), S_minus_F_D1=st(d1S - d1F), S_minus_F_D2=st(d2S - d2F), ceil_S=cS, ceil_F=cF,
                   S_D1_priv=ws(d1S, W["private"]), S_D2_priv=ws(d2S, W["private"]))
print(f"REAL on the STALE pool S: D1 {st(d1S)[0]:+.5f} +- {st(d1S)[1]:.5f} (z {st(d1S)[2]:+.2f}) | D2 {st(d2S)[0]:+.5f} +- {st(d2S)[1]:.5f} (z {st(d2S)[2]:+.2f}) | in-sample D1 ceiling {cS:+.5f} | "
      f"pmix private D1 {res['real']['S_D1_priv'][0]:+.5f}+-{res['real']['S_D1_priv'][1]:.5f} D2 {res['real']['S_D2_priv'][0]:+.5f}+-{res['real']['S_D2_priv'][1]:.5f}")
print(f"REAL on the FRESH train-only pool F: D1 {st(d1F)[0]:+.5f} +- {st(d1F)[1]:.5f} | D2 {st(d2F)[0]:+.5f} +- {st(d2F)[1]:.5f} | in-sample D1 ceiling {cF:+.5f}")
print(f"S minus F (paired): D1 {st(d1S - d1F)[0]:+.5f} +- {st(d1S - d1F)[1]:.5f} (z {st(d1S - d1F)[2]:+.2f}) | D2 {st(d2S - d2F)[0]:+.5f} +- {st(d2S - d2F)[1]:.5f} (z {st(d2S - d2F)[2]:+.2f})", flush=True)
sh = np.bincount(V.fid, madecold, V.nf) / V.M; q = np.digitize(sh, np.quantile(sh, [1 / 3, 2 / 3]))
res["slices"] = {}
for k in range(3):
    m = q == k
    res["slices"][k] = dict(n=int(m.sum()), share=float(sh[m].mean()), loss=st((nS - nF)[m]), D1=st(d1S[m]), D2=st(d2S[m]))
    print(f"  feeds tertile {k + 1} by share of candidates made cold (mean {100 * sh[m].mean():.1f}%) [{int(m.sum())}]: staleness loss {st((nS - nF)[m])[0]:+.5f} | D1 on S {st(d1S[m])[0]:+.5f}+-{st(d1S[m])[1]:.5f} | D2 on S {st(d2S[m])[0]:+.5f}+-{st(d2S[m])[1]:.5f}")
grp = V.fid * 2 + V.cold.astype(np.int64)
res["placebo"] = []
for seed in range(500, 510):
    u = np.random.default_rng(seed).random(V.N)
    so = np.argsort(grp, kind="stable"); sr = np.lexsort((u, grp)); src = np.empty(V.N, np.int64); src[so] = sr
    p1, p2, _ = read(C[src], S, nS)
    res["placebo"].append([st(p1)[0], st(p2)[0]])
    print(f"[WF on S] seed {seed}: D1 {st(p1)[0]:+.5f} D2 {st(p2)[0]:+.5f} | {time.time() - t0:.0f}s", flush=True)
pp = np.array(res["placebo"])
for i, nm in enumerate(("D1", "D2")):
    real = res["real"][f"S_{nm}"][0]
    print(f"placebo {nm} on S: mean {pp[:, i].mean():+.5f} sd {pp[:, i].std(ddof=1):.5f} min {pp[:, i].min():+.5f} max {pp[:, i].max():+.5f} | real {real:+.5f} above {(real > pp[:, i]).sum()} of {len(pp)}")
json.dump(res, open(f"{OUT}/r9_stale.json", "w"), indent=1)
print(f"DONE {time.time() - t0:.0f}s")
