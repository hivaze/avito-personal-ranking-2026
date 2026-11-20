"""Build the top-3 public-probe submissions from the design arrays (out/cand_<name>_{val,test}.npy):
  preds/candP_<name>_{val,test}.npy (new files; spec names for presub_gate) and subs/cand_P_<file>.parquet via src/make_sub.make_sub.
"""
import sys, shutil, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, "<repo>/src")
from make_sub import make_sub
from common import feed_meta
from stack_val import feed_rank_z
_, Mt, ot = feed_meta('test')
ROOT = Path("<repo>"); OUT = ROOT / "nn_x/audit/publicprobe/out"
def detie(x, M, off, eps=4e-6):
    """Break EXACT within-feed ties deterministically (k-th duplicate + k*eps, eps above float32 resolution of z-scores);
    non-tied values are untouched. Needed because make_sub (random key) and presub_gate check B (index order) break exact
    ties differently, which failed B on one M~46 public feed with two identically scored candidates."""
    x = np.asarray(x, np.float64).copy(); n_tie = 0
    for f in range(len(M)):
        o, m = off[f], M[f]; v = x[o:o + m]; order = np.argsort(v, kind="stable"); sv = v[order]
        k = 0
        for i in range(1, m):
            if sv[i] == sv[i - 1]:
                k += 1; v[order[i]] += k * eps; n_tie += 1
            else:
                k = 0
    return x.astype(np.float32), n_tie
TOP3 = {"wft100_FTonly": "cand_P_G3_FTonly", "msplit_M50": "cand_P_G3_msplit_TOlt50", "NEW12": "cand_P_NEW12_memeq"}
for name, fn in TOP3.items():
    for s in ["val", "test"]:
        dst = ROOT / "preds" / f"candP_{name}_{s}.npy"
        if not dst.exists():
            shutil.copy(OUT / f"cand_{name}_{s}.npy", dst)
    tp = ROOT / "preds" / f"candP_{name}_test.npy"; xt, nt = detie(np.load(tp), Mt, ot); np.save(tp, xt); print(f"{name}: {nt} exact within-feed ties broken on test")
    sub = ROOT / "subs" / f"{fn}.parquet"
    # build from exactly the gate's transform (within-feed z, float32) of the stored preds array, so check B is exact
    make_sub(feed_rank_z(ot, Mt, np.load(ROOT / 'preds' / f'candP_{name}_test.npy').astype(np.float32))[:, 1], str(sub))
    print(fn, hashlib.sha256(sub.read_bytes()).hexdigest(), sub.stat().st_size)
