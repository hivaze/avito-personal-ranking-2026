"""Bootstrap selection optimism of picking the best of the 6 pre-declared designs D0..D5 (exp8) on val.
In-bag: choose argmax mean per-feed NDCG on a bootstrap resample of feeds; out-of-bag: its delta vs D0 on the left-out feeds."""
import json, sys
import numpy as np
sys.path.insert(0, "<repo>/solvers/ensemble")
from ens_common import ROOT, Split
S = Split("val")
names = ["D0_v11", "D1_eq_F5_F5cd", "D2_eq_F5_V13", "D3_eq_F5_F5cd_V13", "D4_member_eq15", "D5_eq_F5_F5cd_V13_F4"]
P = np.stack([S.per_feed(np.load(f"{ROOT}/solvers/out/ensemble/cand_{n}_val.npy")) for n in names])
rng = np.random.default_rng(0)
ins, oob, picks = [], [], []
for _ in range(2000):
    b = rng.integers(0, S.n, S.n)
    m = np.ones(S.n, bool); m[b] = False
    j = int(np.argmax(P[:, b].mean(1)))
    picks.append(j)
    ins.append(P[j, b].mean() - P[0, b].mean())
    oob.append(P[j, m].mean() - P[0, m].mean())
ins, oob = np.array(ins), np.array(oob)
print("full-val deltas vs D0:", {n: round(float(P[i].mean() - P[0].mean()), 5) for i, n in enumerate(names)})
print(f"pick frequency: {dict(zip(names, np.bincount(picks, minlength=len(names)).tolist()))}")
print(f"in-bag delta of the pick {ins.mean():+.5f}, out-of-bag {oob.mean():+.5f}, optimism {ins.mean() - oob.mean():+.5f}; P(oob<0) {np.mean(oob < 0):.3f}")
json.dump({"ins": float(ins.mean()), "oob": float(oob.mean()), "picks": np.bincount(picks, minlength=len(names)).tolist(), "p_oob_neg": float(np.mean(oob < 0))},
          open(f"{ROOT}/solvers/out/ensemble/exp10.json", "w"))
