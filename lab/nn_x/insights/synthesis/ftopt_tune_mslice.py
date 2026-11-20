"""Synthesis check (CPU, TUNE half only; CONFIRM labels untouched): does any FT-study trial remove the M<50 FT harm?

For each finished nn_x/ftopt trial with OOF preds for all 4 set-A members:
  delta(trial) = NDCG(pool(trial FT)) - NDCG(pool(n2cpv3 FT)), 0.5/0.5 FT/TO pool as in the study, sliced by M.
Also the FT-half effect on TUNE (pool(n2cpv3) - TO-only pool) and the M-split blend (TO-only for M<50) by M slice.
Mechanism under test (ensemble E2): n2cpv3 uses loss_norm 'batch' (4.9x per-feed weight on M~300 vs M<=64);
trial 8 = n2cpv3 with loss_norm 'feed' only.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "<repo>/nn_x/ftopt")
from ftlib import PREDS, SET_A, Val  # noqa: E402

FT = Path("<repo>/nn_x/ftopt/out")
V = Val()
T = V.tune
to_z = V.zmean([np.load(PREDS / f"nn_{m}_trainonly_val.npy") for m in SET_A])
base_ft = [np.load(PREDS / f"nn_{m}_n2cpv3_val_oof.npy") for m in SET_A]
nd_base = V.nd(V.pool(base_ft, to_z))
nd_to = V.nd(to_z)
small = V.M < 50


def show(name, d):
    out = [name]
    for lab, mk in (("all", T), ("M<50", T & small), ("M>=50", T & ~small)):
        x = d[mk]
        out.append(f"{lab} {x.mean():+.5f}+-{x.std(ddof=1) / np.sqrt(len(x)):.5f}")
    print("  ".join(out))


print(f"TUNE feeds {T.sum()} (M<50 {(T & small).sum()}); CONFIRM not read")
show("FT half on TUNE: pool(n2cpv3) - TO-only pool       ", nd_base - nd_to)
msplit = np.where(small[V.fid], to_z, V.pool(base_ft, to_z))
show("M-split (TO-only for M<50) - pool(n2cpv3)          ", V.nd(msplit) - nd_base)
for t in sorted(int(p.name[1:]) for p in FT.glob("t*") if p.name[1:].isdigit()):
    fs = [FT / f"t{t}" / f"nn_{m}_t{t}_val_oof.npy" for m in SET_A]
    if not all(f.exists() for f in fs):
        continue
    ft = [np.load(f) for f in fs]
    show(f"trial {t:2d}: pool(trial) - pool(n2cpv3)               ", V.nd(V.pool(ft, to_z)) - nd_base)
