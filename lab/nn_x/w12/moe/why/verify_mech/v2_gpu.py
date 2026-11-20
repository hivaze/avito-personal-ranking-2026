"""verify_mech v2 (GPU, read-only): does the arms' VAL deficit track the train-fit excess (mech M6)?
 (a) VAL NDCG of every screen member and its same-seed reference at matched steps @70000 / @77500 / @last (the @last pass is checked
     bit-exactly against the stored @last val preds) -> arm - ref member deltas per step, paired per feed; ref seeds 955-958 added as
     a seed null (ref{951-954} vs ref{955-958} member mean at each step).
 (b) fit on 30,000 PRODUCTION training feeds (p_random <= 0.5, i.e. NOT in the cooldown exploration pool that is replayed during the
     cooldown; holdout excluded; any M) at @70000 and @last -> arm - ref, paired per feed. mech's train-fit feeds (p_random > 0.9,
     M >= 50) are all inside that replayed pool.
Writes verify_mech/v2_ndcg.npz and prints the table."""
import sys, time
import numpy as np, torch
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import *  # noqa  (read-only loaders; OUT of mech is not written)
HERE = Path('<repo>/lab/nn_x/w12/moe/why/verify_mech')
dev = torch.device('cuda:0')
ARMS = dict(ref='kd05w192', A='kd05w192moeffn', B='kd05w192moevo', C='kd05w192b3ff1')
SUF = ('@70000', '@77500', '@last')
m0, cfg = build('w5_e10_kd05w192_s951', dev)
va = load_val(dev, cfg['xt_names'])
f, hold, p = holdout_feeds()
cand = np.setdiff1d(np.where(p <= 0.5)[0], hold)
prod_idx = np.sort(np.random.default_rng(778).choice(cand, 30000, replace=False))
pr = load_hold(dev, cfg['xt_names'], feeds=(f, prod_idx, p))
print(f'[prod] M mean {pr.M_np.mean():.1f} median {np.median(pr.M_np):.0f}; share M>=50 {(pr.M_np >= 50).mean():.3f}; '
      f'share with a positive {np.mean(np.add.reduceat(pr.target_np, pr.cand_off_np) > 0):.3f}', flush=True)
out = {}
t0 = time.time()
jobs = [(g, s) for g in ARMS for s in (951, 952, 953, 954)] + [('ref', s) for s in (955, 956, 957, 958)]
for g, s in jobs:
    v = f'w5_e10_{ARMS[g]}_s{s}'
    for suf in SUF:
        m, _ = build(v, dev, suf)
        pv = predict(m, va, tf32=True)
        if suf == '@last':
            sv, _ = pred_paths(v, '@last')
            dmax = float(np.abs(pv - np.load(sv)).max())
            assert dmax == 0.0, f'{v} val @last parity {dmax}'
        out[f'{g}.{s}{suf}.val'] = split_ndcg(va, pv).astype(np.float32)
        if suf != '@77500' and s <= 954:
            out[f'{g}.{s}{suf}.prod'] = split_ndcg(pr, predict(m, pr)).astype(np.float32)
        del m
    print(f'{v}: ' + ' | '.join(f'{suf} val {out[f"{g}.{s}{suf}.val"].mean():.5f}' + (f' prod {out[f"{g}.{s}{suf}.prod"].mean():.5f}' if f'{g}.{s}{suf}.prod' in out else '')
                                for suf in SUF) + f'  ({time.time() - t0:.0f}s)', flush=True)
np.savez_compressed(HERE / 'v2_ndcg.npz', **out)
S4 = (951, 952, 953, 954)
for part in ('val', 'prod'):
    print(f'== {part}: arm - ref member deltas (same seeds, per-feed mean over 4 seeds; +- feed SE; seed-level SE in brackets)')
    for suf in SUF:
        if f'ref.951{suf}.{part}' not in out:
            continue
        line = f'  {suf:7s}'
        for g in 'ABC':
            per = np.array([float((out[f'{g}.{s}{suf}.{part}'].astype(np.float64) - out[f'ref.{s}{suf}.{part}']).mean()) for s in S4])
            d = np.mean([out[f'{g}.{s}{suf}.{part}'].astype(np.float64) - out[f'ref.{s}{suf}.{part}'] for s in S4], 0)
            line += f' | {g} {d.mean():+.5f}+-{se(d):.5f} [{per.std(ddof=1) / 2:.5f}]'
        if part == 'val':
            dn = np.mean([out[f'ref.{s}{suf}.val'].astype(np.float64) for s in S4], 0) - np.mean([out[f'ref.{s}{suf}.val'].astype(np.float64) for s in (955, 956, 957, 958)], 0)
            line += f' | NULL ref951-954 - ref955-958 {dn.mean():+.5f}+-{se(dn):.5f}'
        print(line, flush=True)
    print(f'== {part}: cooldown effect @last - @70000 (member mean, paired per feed)')
    for g in ARMS:
        d = np.mean([out[f'{g}.{s}@last.{part}'].astype(np.float64) - out[f'{g}.{s}@70000.{part}'] for s in S4], 0)
        print(f'  {g:4s} {d.mean():+.5f}+-{se(d):.5f}', flush=True)
