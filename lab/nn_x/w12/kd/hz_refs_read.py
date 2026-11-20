"""Reference-robustness read of a re-cool arm (information next to RULE hz; witness report 6.3 showed that the ref pool is a low draw under stale inputs).
  hz_refs_read.py <arm> [<arm2> ...]      16-pool of each arm (K + L snapshots, seeds 951-958) minus each of: ref | ref2 | orig (the original wide students) | all48 (ref + ref2 + orig)
Columns per checkpoint kind (selected, @last): val, val M<50, stale-val (GPU files nn_x/w5/preds/stale), SV2 = per-feed mean of the val and stale-val deltas, pmix-private, in-period holdout (reported).
Also prints arm-vs-arm contrasts when several arms are given. CPU, read-only."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition'); sys.path.insert(0, '<repo>/src')
from pc_common import *
import pmix
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh); wts = pmix.weights(); big = m.Mv >= 50
S8 = range(951, 959)
from pathlib import Path
SD = Path(os.environ['STALE_DIR']) if os.environ.get('STALE_DIR') else W / 'stale'   # STALE_DIR=nn_x/w12/stale/hz2n/preds_stale_full reads the like-for-like scenario both_pre2full
KINDS = tuple(os.environ.get('KINDS', ',@last').split(','))
def names(a):
    if a == 'orig': return [f'w5_e10_{f}_s{x}' for f in ('kd05w192', 'kd05t25w192') for x in S8]
    if a == 'all48': return names('ref') + names('ref2') + names('orig')
    return [f'w5_{st}cd_{a}_s{x}' for st in 'kl' for x in S8]
def lv(a, kind):
    N = names(a)
    fv = lambda v: W / f'nn_{v}{kind}_trainonly_val.npy'; fh = lambda v: W / f'holdout/nn_{v}{kind}_holdout.npy'; fs = lambda v: SD / f'nn_{v}{kind}_staleval.npy'
    miss = [v for v in N if not fv(v).exists()]
    if miss: return None
    pool = lambda sp, fn: np.mean([m.z(sp, np.load(fn(v)).astype(np.float32)) for v in N], 0)
    out = {'val': ev(pool('val', fv)), 'hold': eh(pool('hold', fh)) if all(fh(v).exists() for v in N) else None,
           'stale': ev(pool('val', fs)) if all(fs(v).exists() for v in N) else None}
    return out
arms = sys.argv[1:]; refs = ['ref', 'ref2', 'orig', 'all48']
f2 = lambda d: f'{d.mean():+.5f}+-{se(d):.5f} (z {d.mean() / se(d):+.1f})'
for kind in KINDS:
    L = {a: lv(a, kind) for a in arms + refs}
    print(f'=== [{SD.name}] {kind or "selected"} checkpoints: levels val / stale-val: ' + ' | '.join(f'{a} {L[a]["val"].mean():.5f} / {(L[a]["stale"].mean() if L[a]["stale"] is not None else float("nan")):.5f}' for a in L if L[a]))
    pairs = [(a, r) for a in arms for r in refs] + [(a, b) for i, a in enumerate(arms) for b in arms[:i]]
    for a, r in pairs:
        if not L[a] or not L[r]: print(f'  {a} - {r}: files missing'); continue
        dv = L[a]['val'] - L[r]['val']; s = f'  {a:5s} - {r:6s}: val {f2(dv)} M<50 {f2(dv[~big])}'
        if L[a]['stale'] is not None and L[r]['stale'] is not None:
            ds = L[a]['stale'] - L[r]['stale']; sv2 = 0.5 * (dv + ds); s += f' | stale-val {f2(ds)} | SV2 {f2(sv2)} | SVpriv(0.27/0.73) {f2(0.27 * dv + 0.73 * ds)}'
        else: s += ' | stale-val: files missing'
        pr, sr = pmix.wmean_se(dv, wts['private']); pu, su = pmix.wmean_se(dv, wts['public']); s += f' | pmix-public {pu:+.5f}+-{su:.5f} pmix-private {pr:+.5f}+-{sr:.5f}'
        if L[a]['hold'] is not None and L[r]['hold'] is not None: dh = L[a]['hold'] - L[r]['hold']; s += f' | holdout (reported) {f2(dh)}'
        print(s)
