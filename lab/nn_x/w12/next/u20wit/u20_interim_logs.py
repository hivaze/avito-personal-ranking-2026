"""u20wit: INFORMATION ONLY, not the declared read. Training-log evals of the U20 arms (GPU, pooled over user folds: all 30,000 holdout feeds / all val feeds).
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/u20_interim_logs.py
Plateau comparison at MATCHED EPOCH: u20a has 7,871 steps per epoch, u20b / u20c 9,835 (ratio 0.8003), so A at step N is paired with B / C at step 1.25 N
(the same number of visits of every labelled feed; B additionally saw its unlabelled fold). Only steps before A's cooldown (62,968) are used. Single evals of
running models: seed noise about 0.001 each, successive evals of one run are correlated, no per-feed pairing is possible from a log."""
import re, sys, numpy as np
from pathlib import Path
L = Path('<repo>/nn_x/w5/logs'); pat = re.compile(r'^step (\d+)/(\d+) loss [\d.]+ val_ndcg ([\d.]+) holdout_ndcg ([\d.]+)')
def read(v):
    p = L / f'{v}.log'; out = {}
    if not p.exists(): return out, None
    tot = None
    for l in open(p):
        m = pat.match(l)
        if m: out[int(m.group(1))] = (float(m.group(3)), float(m.group(4))); tot = int(m.group(2))
    return out, tot
for s in (951, 952):
    R = {arm: read(f'w5_e10_u20{arm}_s{s}') for arm in 'abc'}
    print(f'seed {s}: last logged step ' + ', '.join(f'u20{arm} {max(R[arm][0]) if R[arm][0] else None}/{R[arm][1]}' for arm in 'abc') + ' | finished (json): ' + ', '.join(f'u20{arm} {(L / f"w5_e10_u20{arm}_s{s}.json").exists()}' for arm in 'abc'))
    rows = []
    for na in range(10000, 62500, 10000):
        nb = int(na * 1.25)
        if na in R['a'][0] and nb in R['b'][0] and nb in R['c'][0]:
            a_, b_, c_ = R['a'][0][na], R['b'][0][nb], R['c'][0][nb]; rows.append((b_[0] - a_[0], b_[1] - a_[1], c_[0] - a_[0], c_[1] - a_[1]))
            print(f'   A step {na:6d} / B, C step {nb:6d}: val A {a_[0]:.5f} B {b_[0]:.5f} C {c_[0]:.5f} | holdout A {a_[1]:.5f} B {b_[1]:.5f} C {c_[1]:.5f} | B-A {b_[0] - a_[0]:+.5f} / {b_[1] - a_[1]:+.5f}  C-A {c_[0] - a_[0]:+.5f} / {c_[1] - a_[1]:+.5f}')
    if rows:
        r = np.array(rows); print(f'   mean of {len(rows)} matched-epoch plateau evals (val / holdout): B-A {r[:, 0].mean():+.5f} / {r[:, 1].mean():+.5f}   C-A {r[:, 2].mean():+.5f} / {r[:, 3].mean():+.5f}   (sd across evals: {r[:, 0].std():.5f} / {r[:, 1].std():.5f}, {r[:, 2].std():.5f} / {r[:, 3].std():.5f})')
