"""07c: terminal-click arms, seeds 1-3: A5 = g_noend offset + (-0.5) on the last shown slot (a NEGATIVE offset = the
terminal positive gets a stronger pull), A6 = the -0.5 last-slot offset alone. A0 re-run for pairing.
CPU GBDT proxy of the proposed EXPLORATION-ONLY examination offset (train-only position offsets on random-order
feeds, where shown position is independent of the item, so no confounding). LightGBM + listwise softmax CE
(proxy_lib_ll), trained on 64K pool feeds (random-like, M>=50, holdout excluded) per seed, 3 seeds, 500 rounds.
Arms (offset = init_score in training only; prediction never sees position):
  A0 none | A1 g(shown rank)+h(rank from end) (joint fit, out/pool_pos_offset.npz) | A2 g only (fit on non-end slots)
  A3 h only (end-of-feed term) | A4 learned: shown rank + rank-from-end as train-only features, fixed at
  inference (rank 20, rank-from-end 50), the eda_labels_position g2 design.
Paired per-feed deltas vs A0 on val (all / M<50 / M>=50) and on the 30K holdout. Writes out/07c_gbdt_terminal.txt"""
import sys, time
import numpy as np
from proxy_lib_ll import *

OUT = "<repo>/nn_x/insights/labels_loss/out"
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

X, y, M = load("pool"); aux = np.load(f"{D}/pool_aux.npy")
Xv, yv, Mv = load("val"); Xv = np.asarray(Xv)
Xh, yh, Mh = load("hold"); Xh = np.asarray(Xh)
o5 = np.load(f"{OUT}/pool_pos_offset.npz"); G, Hh = o5["g_rank"], o5["h_rkend"]
EMAX = len(Hh)
rk_all, rke_all = aux[0].astype(np.int64), aux[1].astype(np.int64)
# A2: g refit on non-end slots only (rank-from-end >= 3), smoothed beyond rank 30
yy = y.astype(np.float64); m = rke_all >= 3
base = yy[m].mean(); g2 = np.zeros(300)
for r in range(300):
    mm = m & (np.minimum(rk_all, 120) == min(r, 120))
    g2[r] = np.log(max(yy[mm].sum(), 1) / max(mm.sum() * base, 1e-9))
g2s = g2.copy()
for r in range(30, 300):
    g2s[r] = g2[max(30, r - 10):min(299, r + 10) + 1].mean()
P("A2 g (non-end slots) at ranks 0,1,2,5,10,20,50,100: " + " ".join(f"{g2s[r]:+.2f}" for r in [0, 1, 2, 5, 10, 20, 50, 100]))

def offset(kind, rk, rke):
    gg = G[np.minimum(rk, 299)]; hh = np.where(rke < EMAX, Hh[np.minimum(rke, EMAX - 1)], 0.0)
    if kind == "A1": return gg + hh
    if kind == "A2": return g2s[np.minimum(rk, 299)]
    if kind == "A5": return g2s[np.minimum(rk, 299)] - 0.5 * (rke == 0)  # g + terminal-click UP-weighting
    if kind == "A6": return -0.5 * (rke == 0).astype(np.float64)  # terminal-click up-weighting only
    if kind == "A2s": return g2s[np.minimum(rk, 299)] * (rke >= 3)  # g on non-end slots only, end slots untouched
    if kind == "A3": return hh
    return None

ARMS = ["A0", "A5", "A6"]
STEPS = [100, 200, 300, 400, 500]
ov, oh = offs(Mv), offs(Mh)
res = {a: {"val": [], "hold": []} for a in ARMS}
for seed in [1, 2, 3]:
    rng = np.random.default_rng(seed)
    fe = np.sort(rng.choice(len(M), 64000, replace=False))
    Xs, ys, Ms, r = subset(X, y, M, fe)
    rk, rke = rk_all[r], rke_all[r]
    for a in ARMS:
        t0 = time.time()
        if a == "A4":
            Xa = np.hstack([Xs, np.minimum(rk, 299)[:, None].astype(np.float32), np.minimum(rke, 299)[:, None].astype(np.float32)])
            bst = train(Xa, ys, Ms, rounds=500, seed=seed)
            Xva = np.hstack([Xv, np.full((len(Xv), 1), 20, np.float32), np.full((len(Xv), 1), 50, np.float32)])
            Xha = np.hstack([Xh, np.full((len(Xh), 1), 20, np.float32), np.full((len(Xh), 1), 50, np.float32)])
        else:
            init = offset(a, rk, rke)
            bst = train(Xs, ys, Ms, init=init, rounds=500, seed=seed)
            Xva, Xha = Xv, Xh
        cv = ndcg_curve(bst, Xva, yv, Mv, STEPS); ch = ndcg_curve(bst, Xha, yh, Mh, [500])
        res[a]["val"].append(cv); res[a]["hold"].append(ch[500])
        P(f"seed {seed} {a}: val " + " ".join(f"{k}:{v.mean():.5f}" for k, v in cv.items()) +
          f" | hold@500 {ch[500].mean():.5f} ({time.time()-t0:.0f}s)")

P("\nseed-averaged paired deltas vs A0 (SE of the seed-averaged per-feed delta)")
short = Mv < 50
for a in ARMS[1:]:
    for k in [300, 500]:
        d = np.mean([res[a]["val"][i][k] - res["A0"]["val"][i][k] for i in range(len(res['A0']['val']))], axis=0)
        P(f"{a} @{k}: val {d.mean():+.5f} ({d.std()/np.sqrt(len(d)):.5f})  M<50 {d[short].mean():+.5f} "
          f"({d[short].std()/np.sqrt(short.sum()):.5f})  M>=50 {d[~short].mean():+.5f} ({d[~short].std()/np.sqrt((~short).sum()):.5f})"
          f"  per-seed " + " ".join(f"{(res[a]['val'][i][k]-res['A0']['val'][i][k]).mean():+.5f}" for i in range(len(res['A0']['val']))))
    dh = np.mean([res[a]["hold"][i] - res["A0"]["hold"][i] for i in range(len(res['A0']['val']))], axis=0)
    P(f"{a} hold@500: {dh.mean():+.5f} ({dh.std()/np.sqrt(len(dh)):.5f})")
P("A0 levels: val@500 " + " ".join(f"{res['A0']['val'][i][500].mean():.5f}" for i in range(len(res['A0']['val']))))
open(f"{OUT}/07c_gbdt_terminal.txt", "w").write("\n".join(lines) + "\n")
np.savez_compressed(f"{OUT}/07c_perfeed.npz", **{f"{a}_{k}_s{i}": res[a]["val"][i][k] for a in ARMS for k in [300, 500] for i in range(3)}, **{f"{a}_hold_s{i}": res[a]["hold"][i] for a in ARMS for i in range(3)})
