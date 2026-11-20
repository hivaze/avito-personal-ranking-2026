"""Step 1: halves, temperature on half A, actual + re-drawn per-positive losses, for the final score and each branch.

Out: data/setup_<name>.npz with T, p, E (expected positives per row), L_act, redraw rows/losses.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *  # noqa: F401,F403

R = int(os.environ.get("R", 200))
t0 = time.time()
F, M, off = layout()
half = halves(F)
np.save(HERE / "data/half.npy", half)
fid = np.repeat(np.arange(len(M)), M)
D = load_table(["score", "s_ft", "s_to", "label"])
y = D["label"].to_numpy().astype(np.float64)
npos = np.bincount(fid, y, len(M)).astype(np.int64)
print(f"feeds {len(M)}  half A {int((half == 0).sum())} B {int((half == 1).sum())}  positives {int(y.sum())}")
for name, col in [("final", "score"), ("ft", "s_ft"), ("to", "s_to")]:
    s = D[col].to_numpy().astype(np.float64)
    T = fit_T(s, y, off, M, half == 0)
    TB = fit_T(s, y, off, M, half == 1)
    x = s / T
    p = softmax_p(x, off, M)
    rk = model_rank(s, off, M)
    L = actual_pos_loss(y, rk, off, M, DISC)
    nd = ndcg_per_feed(s, y, off, M)
    assert np.allclose(np.bincount(fid, L, len(M)), 1 - nd, atol=1e-9), "loss split must equal 1 - NDCG"
    rows, loss = redraw(x, rk, off, M, npos, R, 17, DISC)
    # expected positives per row: exact for 1-positive feeds, Plackett-Luce inclusion frequency otherwise
    cnt = np.bincount(rows.ravel(), minlength=len(s)) / R
    E = np.where(npos[fid] == 1, p, cnt)
    nd_rd = 1 - loss.sum(1) / len(M)
    print(f"[{name}] T_A {T:.4f} (T_B {TB:.4f})  NDCG {nd.mean():.5f}  loss {L.sum():.1f}  "
          f"redraw NDCG {nd_rd.mean():.5f} ± {nd_rd.std():.5f} (over {R} draws)  "
          f"NLL/pos A {nll(x, y, off, M, half == 0) / y[half[fid] == 0].sum():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    for h in (0, 1):
        fm = half == h
        print(f"   half {'AB'[h]}: NDCG {nd[fm].mean():.5f}  redraw {(1 - loss[:, half[fid[rows[0]]] == h].sum(1) / fm.sum()).mean():.5f}")
    np.savez(HERE / f"data/setup_{name}.npz", T=T, TB=TB, p=p, E=E, L=L, rows=rows.astype(np.int32),
             loss=loss.astype(np.float32), rk=rk.astype(np.int16), nd=nd)
print("done", f"{time.time() - t0:.0f}s")
