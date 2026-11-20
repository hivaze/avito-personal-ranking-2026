"""A6: pairwise member correlation (val, within-feed z of train-only preds) by family: does m02 make members alike?"""
import os, sys, itertools
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(k, "12")
sys.path.insert(0, "<repo>/nn_x/pubdiag/overfit")
import numpy as np
from lib import Split, ROOT
S = Split("val")
def z(m):
    for d in (ROOT / "preds", ROOT / "nn_x/r6/preds"):
        f = d / f"nn_{m}_trainonly_val.npy"
        if f.exists():
            return S.z(np.load(f))
    return None
fams = {
    "F7 mix.1 (scratch)": [f"v17_pre2s_s{s}" for s in range(101, 108)],
    "F7b c0 re-cool": [f"r6_cd_c0_r{s}" for s in range(101, 108)],
    "F7x m02 re-cool": [f"r6_cd_m02_r{s}" for s in range(101, 108)],
    "F7x70 m02 re-cool": [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)],
    "F9 lr1 mix.1": [f"v22_pre2s_lr1_s{s}" for s in range(401, 407)],
    "F13l lr1 m02": [f"v28_lr1_m02_s{s}" for s in range(411, 418)],
    "F10 GRU mix.1": [f"v23_gru_pre2s_s{s}" for s in range(501, 507)],
    "F13g GRU m02": [f"v27_gru_m02_s{s}" for s in (511, 513, 515)],
    "F12 d192 mix.1": [f"v26_d192_pre2s_s{s}" for s in range(701, 708)],
    "d192 m02 (v29)": [f"v29_d192_m02_s{s}" for s in (711, 712, 713)],
    "F7 m02 scratch (v30)": [f"v30_f7_m02_s{s}" for s in range(901, 907)],
}
for f, ms in fams.items():
    Zs = {m: z(m) for m in ms}
    Zs = {m: v for m, v in Zs.items() if v is not None}
    cs = [np.corrcoef(Zs[a], Zs[b])[0, 1] for a, b in itertools.combinations(Zs, 2)]
    if cs:
        print(f"{f:24s} members {len(Zs)} pairs {len(cs):2d} mean corr {np.mean(cs):.4f} +- {np.std(cs, ddof=1) / np.sqrt(len(cs)) if len(cs) > 1 else 0:.4f}")
    else:
        print(f"{f:24s} no preds yet")
