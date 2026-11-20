"""Step 6 (ii), label-free: does the count distribution differ between val, public-like and private-like rows?
PSI of the raw count (bins 0,1,2,3-4,5-9,10-19,20+) and of the within-feed z / rank forms (10 val-quantile bins), per window,
all / cold / warm; per-day mean and zero share. Flag (DECLARATION G2): PSI >= 0.25.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 NUMBA_NUM_THREADS=6 .venv/bin/python nn_x/w12/fresh/pregate/s6_transfer.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
from lib import OUT, WIN, forms, load_union, psi, psi_q

F, R = load_union()
cm = np.load(f"{OUT}/cnt_main.npy"); meta = np.load(f"{OUT}/rowmeta.npz")
part, cold, day = meta["part"], meta["cold"], meta["day"]
res = {}
print("PSI of the RAW count: val -> public-like / val -> private-like / public-like -> private-like   (flag >= 0.25)")
print(f"{'window':6s} {'subset':5s} {'val->pub':>9s} {'val->priv':>10s} {'pub->priv':>10s} | zero share val / pub / priv | mean val / pub / priv")
for w, wn in enumerate(WIN):
    for sub, m in (("all", np.ones(len(cold), bool)), ("cold", cold), ("warm", ~cold)):
        x = [cm[(part == i) & m, w] for i in range(3)]
        p = [psi(x[0], x[1]), psi(x[0], x[2]), psi(x[1], x[2])]
        res[f"raw|{wn}|{sub}"] = dict(psi=p, zero=[float((v == 0).mean()) for v in x], mean=[float(v.mean()) for v in x])
        print(f"{wn:6s} {sub:5s} {p[0]:9.3f} {p[1]:10.3f} {p[2]:10.3f} | " + " / ".join(f"{(v == 0).mean():.3f}" for v in x) + " | " + " / ".join(f"{v.mean():.3f}" for v in x)
              + ("   FLAG" if max(p[:2]) >= 0.25 else ""))
print("\nPSI of the WITHIN-FEED forms (10 val-quantile bins): z of log1p(count), percent rank; val -> public-like / val -> private-like")
for w, wn in enumerate(WIN):
    r, z = forms(F["goff"], F["M"], cm[:, w].astype(np.float64))
    for nm, v in (("z", z), ("rank", r)):
        p = [psi_q(v[part == 0], v[part == 1]), psi_q(v[part == 0], v[part == 2])]
        sd = [float(v[part == i].std()) for i in range(3)]
        res[f"{nm}|{wn}"] = dict(psi=p, sd=sd)
        print(f"{wn:6s} {nm:5s} {p[0]:9.3f} {p[1]:10.3f} | sd val / pub / priv " + " / ".join(f"{s:.3f}" for s in sd) + ("   FLAG" if max(p) >= 0.25 else ""))
    # share of feeds in which the form is constant (no information inside the feed)
    gid = meta["gid"]; ng = len(F["M"])
    mx = np.maximum.reduceat(cm[:, w], F["goff"]); const = mx == 0
    pf = F["split"].astype(int) + (F["split"] == 1) * (F["ts"] >= 1790497920)
    res[f"feeds_all_zero|{wn}"] = [float(const[pf == i].mean()) for i in range(3)]
    print(f"{wn:6s} feeds whose candidates ALL have count 0: val {const[pf == 0].mean():.3f} public-like {const[pf == 1].mean():.3f} private-like {const[pf == 2].mean():.3f}")
print("\nPer Moscow day: mean count (all rows) per window")
for d in (20718, 20720, 20722, 20723, 20724, 20725):
    m = day == d
    res[f"day|{d}"] = [float(cm[m, w].mean()) for w in range(6)]
    print(f"  day {d}: " + " | ".join(f"{wn} {cm[m, w].mean():.3f}" for w, wn in enumerate(WIN)))
print("\nWithin the first / last test day and within val day 1: mean as-of count by third of the day (position inside a day)")
tod = (meta["ts"] + 10800) % 86400
for d in (20718, 20720, 20722, 20725):
    m = day == d
    q = np.quantile(tod[m], [1 / 3, 2 / 3]); t3 = np.digitize(tod[m], q)
    print(f"  day {d}: a share>=1 by third " + " / ".join(f"{100 * (cm[m, 0][t3 == k] >= 1).mean():.1f}%" for k in range(3))
          + " | e (later-only) share>=1 by third " + " / ".join(f"{100 * (cm[m, 5][t3 == k] >= 1).mean():.1f}%" for k in range(3)))
    res[f"third|{d}"] = dict(a=[float((cm[m, 0][t3 == k] >= 1).mean()) for k in range(3)], e=[float((cm[m, 5][t3 == k] >= 1).mean()) for k in range(3)])
json.dump(res, open(f"{OUT}/s6_transfer.json", "w"), indent=1)
print("DONE")
