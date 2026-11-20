"""w12/moe/why/mech: fit gap at MATCHED steps (@70000 = before the WSD cooldown starts at 78,680; @77500; @last = 98,350) for every screen model:
NDCG on 30,000 TRAINING feeds (holdout selection rule, disjoint from the holdout) vs the 30,000 holdout feeds. arm - ref paired per feed."""
import sys
import numpy as np, torch
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import *  # noqa
dev = torch.device('cuda:0')
ARMS = dict(ref='kd05w192', A='kd05w192moeffn', B='kd05w192moevo', C='kd05w192b3ff1')
SEEDS = (951, 952, 953, 954); SUF = ('@70000', '@77500', '@last')
m0, cfg = build('w5_e10_kd05w192_s951', dev)
ho = load_hold(dev, cfg['xt_names'])
tf = load_hold(dev, cfg['xt_names'], feeds=trainfit_feeds())
out = {}
for g, arm in ARMS.items():
    for s in SEEDS:
        for suf in SUF:
            v = f'w5_e10_{arm}_s{s}'
            m, _ = build(v, dev, suf)
            ph = predict(m, ho)
            if suf == '@last':
                _, sp = pred_paths(v, '@last'); assert np.abs(ph - np.load(sp)).max() == 0.0, 'holdout @last parity'
            out[f'{g}.{s}{suf}.hold'] = split_ndcg(ho, ph).astype(np.float32)
            out[f'{g}.{s}{suf}.train'] = split_ndcg(tf, predict(m, tf)).astype(np.float32)
        print(v, ' | '.join(f"{suf} train {out[f'{g}.{s}{suf}.train'].mean():.5f} hold {out[f'{g}.{s}{suf}.hold'].mean():.5f}" for suf in SUF), flush=True)
np.savez_compressed(OUT / 'trainfit2_ndcg.npz', **out)
for suf in SUF:
    print(f'== {suf}')
    for g in ARMS:
        tr = np.mean([out[f'{g}.{s}{suf}.train'] for s in SEEDS], 0); hd = np.mean([out[f'{g}.{s}{suf}.hold'] for s in SEEDS], 0)
        line = f'{g:4s} member-mean train {tr.mean():.5f} hold {hd.mean():.5f} gap {tr.mean() - hd.mean():+.5f}'
        if g != 'ref':
            dt = np.mean([out[f'{g}.{s}{suf}.train'].astype(np.float64) - out[f'ref.{s}{suf}.train'] for s in SEEDS], 0)
            dh = np.mean([out[f'{g}.{s}{suf}.hold'].astype(np.float64) - out[f'ref.{s}{suf}.hold'] for s in SEEDS], 0)
            line += (f' | {g} - ref (paired per feed): train {dt.mean():+.5f}+-{se(dt):.5f} hold {dh.mean():+.5f}+-{se(dh):.5f} '
                     f'gap diff {dt.mean() - dh.mean():+.5f}+-{np.hypot(se(dt), se(dh)):.5f}')
        print(line, flush=True)
print('== cooldown effect (@last - @70000, member mean, paired per feed)')
for g in ARMS:
    dt = np.mean([out[f'{g}.{s}@last.train'].astype(np.float64) - out[f'{g}.{s}@70000.train'] for s in SEEDS], 0)
    dh = np.mean([out[f'{g}.{s}@last.hold'].astype(np.float64) - out[f'{g}.{s}@70000.hold'] for s in SEEDS], 0)
    print(f'{g:4s} train {dt.mean():+.5f}+-{se(dt):.5f} | hold {dh.mean():+.5f}+-{se(dh):.5f}', flush=True)
