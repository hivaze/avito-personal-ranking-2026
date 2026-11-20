"""A8: diversity cost of m02. For each family, the 3-member TO pool gain over its mean member (val, holdout),
averaged over all 3-member subsets (member-equal per-feed z-mean). m02 members correlate more (a6); does the pool gain shrink?"""
import os, sys, itertools
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(k, "12")
sys.path.insert(0, "<repo>/nn_x/pubdiag/overfit")
import numpy as np
from lib import Split, ROOT
SP = {s: Split(s) for s in ("val", "holdout")}
cache = {}
def z(m, s):
    if (m, s) in cache:
        return cache[(m, s)]
    out = None
    if s == "holdout":
        for d in (ROOT / "preds/holdout", ROOT / "nn_x/r6/preds/holdout"):
            f = d / f"nn_{m}_holdout.npy"
            if f.exists():
                out = SP[s].z(np.load(f)); break
    else:
        for d in (ROOT / "preds", ROOT / "nn_x/r6/preds"):
            f = d / f"nn_{m}_trainonly_val.npy"
            if f.exists():
                out = SP[s].z(np.load(f)); break
    cache[(m, s)] = out
    return out
fams = {
    "F7 mix.1": [f"v17_pre2s_s{s}" for s in range(101, 108)],
    "F7x m02 re-cool": [f"r6_cd_m02_r{s}" for s in range(101, 108)],
    "F7 m02 scratch (v30)": [f"v30_f7_m02_s{s}" for s in range(901, 907)],
    "F10 GRU mix.1": [f"v23_gru_pre2s_s{s}" for s in range(501, 507)],
    "F13g GRU m02": [f"v27_gru_m02_s{s}" for s in (511, 513, 515)],
    "F12 d192 mix.1": [f"v26_d192_pre2s_s{s}" for s in range(701, 708)],
    "d192 m02 (v29)": [f"v29_d192_m02_s{s}" for s in (711, 712, 713)],
    "F9 lr1 mix.1": [f"v22_pre2s_lr1_s{s}" for s in range(401, 407)],
    "F13l lr1 m02": [f"v28_lr1_m02_s{s}" for s in range(411, 418)],
}
print("family                   n | val: mean member  3-pool  gain        | holdout: mean member  3-pool  gain")
for f, ms in fams.items():
    out = []
    for s in ("val", "holdout"):
        mm = [m for m in ms if z(m, s) is not None]
        if len(mm) < 3:
            out.append(None); continue
        nd = {m: SP[s].nd(z(m, s)).mean() for m in mm}
        subs = list(itertools.combinations(mm, 3))[:20]
        g = [SP[s].nd(np.mean([z(m, s) for m in c], 0)).mean() - np.mean([nd[m] for m in c]) for c in subs]
        out.append((np.mean(list(nd.values())), np.mean([SP[s].nd(np.mean([z(m, s) for m in c], 0)).mean() for c in subs]), np.mean(g), len(mm)))
    if out[0] is None:
        print(f"{f:24s} too few members"); continue
    v, h = out
    hs = f"{h[0]:.5f} {h[1]:.5f} {h[2]:+.5f}" if h else "n/a"
    print(f"{f:24s} {v[3]} | {v[0]:.5f} {v[1]:.5f} {v[2]:+.5f} | {hs}")
