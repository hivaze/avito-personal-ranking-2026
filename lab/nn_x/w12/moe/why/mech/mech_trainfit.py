"""w12/moe/why/mech: fit gap. NDCG of every screen model on 30,000 TRAINING feeds (holdout selection rule, disjoint from the holdout, seen in
all 10 epochs) vs the 30,000 holdout feeds (never trained on). Per-feed NDCG saved; arm - ref deltas paired per feed (same seed)."""
import json, sys
import numpy as np, torch
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import *  # noqa
dev = torch.device('cuda:0')
ARMS = dict(ref='kd05w192', A='kd05w192moeffn', B='kd05w192moevo', C='kd05w192b3ff1')
SEEDS = (951, 952, 953, 954)
m0, cfg = build('w5_e10_kd05w192_s951', dev)
ho = load_hold(dev, cfg['xt_names'])
tf = load_hold(dev, cfg['xt_names'], feeds=trainfit_feeds())
out = {}
for g, arm in ARMS.items():
    for s in SEEDS:
        v = f'w5_e10_{arm}_s{s}'
        m, _ = build(v, dev)
        out[f'{g}.{s}.hold'] = split_ndcg(ho, predict(m, ho)).astype(np.float32)
        out[f'{g}.{s}.train'] = split_ndcg(tf, predict(m, tf)).astype(np.float32)
        print(v, f"train {out[f'{g}.{s}.train'].mean():.5f} hold {out[f'{g}.{s}.hold'].mean():.5f} gap {out[f'{g}.{s}.train'].mean() - out[f'{g}.{s}.hold'].mean():+.5f}", flush=True)
np.savez_compressed(OUT / 'trainfit_ndcg.npz', **out)
for g in ARMS:
    tr = np.mean([out[f'{g}.{s}.train'] for s in SEEDS], 0); hd = np.mean([out[f'{g}.{s}.hold'] for s in SEEDS], 0)
    print(f'{g}: member-mean train {tr.mean():.5f}+-{se(tr):.5f} hold {hd.mean():.5f}+-{se(hd):.5f} gap {tr.mean() - hd.mean():+.5f}+-{np.hypot(se(tr), se(hd)):.5f}')
    if g != 'ref':
        dt = np.mean([out[f'{g}.{s}.train'].astype(np.float64) - out[f'ref.{s}.train'] for s in SEEDS], 0)
        dh = np.mean([out[f'{g}.{s}.hold'].astype(np.float64) - out[f'ref.{s}.hold'] for s in SEEDS], 0)
        print(f'   {g} - ref (member mean, paired per feed): train {dt.mean():+.5f}+-{se(dt):.5f} | hold {dh.mean():+.5f}+-{se(dh):.5f} | '
              f'gap difference {dt.mean() - dh.mean():+.5f}+-{np.hypot(se(dt), se(dh)):.5f}')
