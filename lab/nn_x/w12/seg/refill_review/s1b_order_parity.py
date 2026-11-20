"""POST-HOC, label-free: (i) refined aligner-free order test (common keys restricted to the deep stored tail, so that a repeated (item, type, source)
whose old occurrence left the window is not counted); (ii) lag-matched coverage of an extended-history arm on train feeds."""
import numpy as np
from collections import Counter
from rr_lib import ROOT, SPLITS, Tables
T = Tables()
hold, _ = T.holdout()
STUDY = f"{ROOT}/nn_x/w12/seg/refill/out"
def take(arr, rows):
    o = np.argsort(rows, kind="stable"); out = np.empty(len(rows), arr.dtype); out[o] = arr[rows[o]]; return out
print("## (i) order test, study's T events, 300 feeds per set (rng 4242); common keys = unique in both AND stored at index >= 200 / >= 128")
for name, s in (("holdout", "train"), ("val", "val"), ("test", "test")):
    z = np.load(f"{STUDY}/align_{name}.npz"); q = z["fid"]; sc0 = SPLITS.index(s)
    pick = np.sort(np.random.default_rng(4242).choice(len(q), 300, replace=False))
    for thr in (200, 128, 0):
        n_ev = n_viol = n_feed = bad_feed = 0; tot_common = tot_deeper = 0
        for i in pick:
            sc, row = z["T_sc"][i].astype(np.int64), z["T_row"][i]; m = sc >= 0
            if not (z["A"][i] and z["B"][i]) or m.sum() == 0: continue
            n_feed += 1
            lk, _, _ = T.hist(sc0, q[i]); cl = Counter(lk.tolist()); lpos = {k: j for j, k in enumerate(lk.tolist())}
            fb = False
            for c in np.unique(sc[m]):
                ss = SPLITS[c]; r = row[m][sc[m] == c]; g = take(T.hf[ss], r).astype(np.int64)
                for gg in np.unique(g):
                    ek, _, _ = T.hist(int(c), int(gg)); ce = Counter(ek.tolist())
                    cp = np.array([j for j, k in enumerate(ek.tolist()) if ce[k] == 1 and cl.get(k, 0) == 1 and lpos[k] >= thr])
                    p = r[g == gg] - T.hoff[ss][gg]
                    if len(cp) == 0: continue
                    deeper = (cp[None, :] > p[:, None]).sum(1)
                    n_ev += len(p); n_viol += int((deeper > 0).sum()); tot_common += len(cp) * len(p); tot_deeper += int(deeper.sum()); fb |= bool((deeper > 0).any())
            bad_feed += fb
        print(f"   {name}, stored index >= {thr}: feeds {n_feed}, T events {n_ev}, with >= 1 such common event deeper in the source snapshot {n_viol} = {100 * n_viol / max(n_ev, 1):.2f}% ({bad_feed} feeds); share of (event, common key) pairs out of order {100 * tot_deeper / max(tot_common, 1):.3f}%")
print("\n## (ii) train feeds (capped, user_id != 0): share with a strictly earlier TRAIN snapshot at a gap >= L days, by train day")
s = "train"; u, t = T.uid[s], T.ts[s]
o = np.lexsort((t, u)); us, ts = u[o], t[o]
ub, st = np.unique(us, return_index=True); tmin = np.minimum.reduceat(ts, st)
cap = (T.H[s] == 256) & (u != 0); first = tmin[np.searchsorted(ub, u)]
day = (t - t.min()) // 86400
for L in (0, 3, 7, 10):
    has = cap & ((t - first) >= max(L * 86400, 1))
    print(f"   gap >= {L:2d} d (to the user's FIRST train snapshot): {100 * has.mean():.2f}% of train feeds; by day: " + " ".join(f"d{d}:{100 * has[day == d].mean():.1f}" for d in np.unique(day) if (day == d).sum() > 1000))
for nm, ss, allowed in (("val", "val", ("train", "val")), ("test", "test", ("train", "val", "test"))):
    uu = np.concatenate([T.uid[a] for a in allowed]); tt = np.concatenate([T.ts[a] for a in allowed])
    oo = np.argsort(uu, kind="stable"); ub2, st2 = np.unique(uu[oo], return_index=True); tm2 = np.minimum.reduceat(tt[oo], st2)
    capv = (T.H[ss] == 256) & (T.uid[ss] != 0); f2 = tm2[np.searchsorted(ub2, T.uid[ss])]
    print(f"   {nm}: " + " | ".join(f"gap >= {L} d: {100 * (capv & ((T.ts[ss] - f2) >= max(L * 86400, 1))).mean():.2f}%" for L in (0, 3, 7, 10)))
