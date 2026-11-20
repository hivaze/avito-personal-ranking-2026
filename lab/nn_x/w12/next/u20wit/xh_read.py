"""u20wit: the INDUCTIVE baseline of the transductive read, CPU vs CPU: B - A (and C - A) on a random subsample of the 24,026 cross-user holdout feeds (XH; xh_build.py, default 8,000) replayed on the CPU,
and the EXCESS of the transductive witness over it: (B - A on W0L / W0S) - (B - A on XH), independent feed sets, SE = sqrt(se1^2 + se2^2).
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/xh_read.py [--seeds 951,952]
Needs cache/scxh/w5_e10_u20{a,b[,c]}_s<seed>__tr.npy (xh_score.py) and cache/sc/..__tr.npy (wit_score.py). Information only; no rule is attached.
XH feeds: of users of folds 1-4, in nobody's training set; A and B both trained on these users' other feeds WITH labels, so B - A here = the effect of the
unlabelled fold on unseen feeds of labelled users (the column the declared rule reads, here on the CPU). Holdout labels are read for evaluation only."""
from wit_common import *
import argparse
from stack_val import feed_rank_z
ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='951,952'); a = ap.parse_args(); seeds = [int(s) for s in a.seeds.split(',')]
name = lambda arm, s: f'w5_e10_u20{arm}_s{s}'
sm = np.load(CACHE / 'subxh_meta.npz'); M = sm['M'].astype(np.int64); off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64); y = np.load(CACHE / 'subxh/cand_target.npy').astype(np.float32)
tm = train_meta(); hold = tm['hold']; Mh = tm['M'][hold]; oh = np.concatenate([[0], np.cumsum(Mh)[:-1]]); ix = np.searchsorted(hold, sm['src_fidx']); assert (hold[ix] == sm['src_fidx']).all() and (tm['xf'][hold[ix]] != 0).all()
hrows = np.concatenate([np.arange(oh[i], oh[i] + Mh[i]) for i in ix]); yh = np.load(W5P / 'holdout/holdout_y.npy'); assert (yh[hrows] == y).all(), 'labels of the XH cache differ from holdout_y'
E = lambda x: ndcg_feeds(x, y, off, M); Z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]
have = {arm: [s for s in seeds if (CACHE / 'scxh' / f'{name(arm, s)}__tr.npy').exists()] for arm in 'abc'}
print(f'XH = cross-user holdout (random subsample of the 24,026 feeds), CPU fp32 replay: {len(M)} feeds, {len(y)} rows; scored: ' + ', '.join(f'u20{arm} {have[arm]}' for arm in 'abc'))
e, eg, zz = {}, {}, {}
for arm in 'abc':
    for s in have[arm]:
        x = np.load(CACHE / 'scxh' / f'{name(arm, s)}__tr.npy'); e[arm, s] = E(x); zz[arm, s] = Z(x)
        eg[arm, s] = E(np.load(W5P / f'holdout/nn_{name(arm, s)}_holdout.npy')[hrows].astype(np.float32))
        print(f'  {name(arm, s)}: CPU {e[arm, s].mean():.5f}  stored GPU file {eg[arm, s].mean():.5f}  CPU - GPU {fmt(e[arm, s] - eg[arm, s])}')
D = {}
for X, Y, lab in (('b', 'a', 'B-A'), ('c', 'a', 'C-A'), ('b', 'c', 'B-C')):
    ss = [s for s in have[X] if s in have[Y]]
    if not ss: continue
    d = np.mean([e[X, s] - e[Y, s] for s in ss], 0); dg = np.mean([eg[X, s] - eg[Y, s] for s in ss], 0); D[lab] = d
    t = f'  {lab} member ({len(ss)} seeds), paired per feed, n {len(d)}: CPU {fmt(d)} [per seed ' + ' '.join(f'{np.mean(e[X, s] - e[Y, s]):+.5f}' for s in ss) + f'] | same feeds, stored GPU files {fmt(dg)} | CPU - GPU of the delta {fmt(d - dg)}'
    if have[X] == have[Y] and len(ss) > 1: t += f' | {len(ss)}-pool CPU {fmt(E(np.mean([zz[X, s] for s in ss], 0)) - E(np.mean([zz[Y, s] for s in ss], 0)))}'
    print(t)
# excess of the transductive witness over the inductive baseline (selected checkpoints, dom tr, CPU vs CPU)
wm = np.load(CACHE / 'sub_meta.npz'); Mw = wm['M'].astype(np.int64); ow = np.concatenate([[0], np.cumsum(Mw)[:-1]]).astype(np.int64); yw = np.load(CACHE / 'sub/cand_target.npy').astype(np.float32)
fidw = np.repeat(np.arange(len(Mw)), Mw); sid = wm['set_id']; Ew = lambda x: ndcg_feeds(np.nan_to_num(x), yw, ow, Mw)
def full(arm, s):
    p_ = CACHE / 'sc' / f'{name(arm, s)}__tr.npy'; return p_.exists() and not np.isnan(np.load(p_)[np.isin(sid, (0, 1, 2))[fidw]]).any()
for X, Y, lab in (('b', 'a', 'B-A'), ('c', 'a', 'C-A')):
    ss = [s for s in seeds if full(X, s) and full(Y, s) and s in have[X] and s in have[Y]]
    if not ss: continue
    dx = np.mean([e[X, s] - e[Y, s] for s in ss], 0); dw = np.mean([Ew(np.load(CACHE / 'sc' / f'{name(X, s)}__tr.npy')) - Ew(np.load(CACHE / 'sc' / f'{name(Y, s)}__tr.npy')) for s in ss], 0)
    print(f'  EXCESS of {lab} over the cross-user holdout (same seeds {ss}; CPU vs CPU; independent feed sets)' + (' [C - A on W0* is IN-SAMPLE for C: bound only]' if lab == 'C-A' else '') + ':')
    for rn, mk in (('W0S short', sid == 0), ('W0L long', sid == 1), ('W0L, old p>0.9 (holdout-like)', (sid == 1) & (wm['pold'] > 0.9)), ('W0S + W0L', (sid == 0) | (sid == 1)), ('H0 fold-0 holdout', sid == 2)):
        print(f'    {rn:30s} n {int(mk.sum()):6d}: {lab} {fmt(dw[mk])}  minus XH {fmt(dx)}  =  {dw[mk].mean() - dx.mean():+.5f}+-{np.sqrt(se(dw[mk]) ** 2 + se(dx) ** 2):.5f}')
print('DONE')
