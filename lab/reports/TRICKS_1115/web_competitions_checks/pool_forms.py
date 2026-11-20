"""Read-only CPU check: pooling form of the 16 wide + 4 popularity-light full-data members on val.
Forms: z-mean (file form), rank-mean, softmax-probability mean (T=1), raw-logit mean. No selection: all four are reported."""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
P = Path('<repo>/preds')
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy')
S8 = range(951, 959); S4 = range(951, 955)
FD16 = [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8]; FDP = [f'abl_fd10_pop1kd05w192_s{s}' for s in S4]
fid = np.repeat(np.arange(len(Mv)), Mv)
def rz(x): return feed_rank_z(ov, Mv, np.asarray(x, np.float32))
def z(x): return rz(x)[:, 1]
def rk(x): return rz(x)[:, 0]
def sm(x):
    x = np.asarray(x, np.float64); mx = np.maximum.reduceat(x, ov); e = np.exp(x - mx[fid]); s = np.add.reduceat(e, ov); return e / s[fid]
FORMS = {'zmean': z, 'rankmean': rk, 'softmaxmean': sm, 'rawmean': lambda x: np.asarray(x, np.float64)}
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float64), yv, ov, Mv)
def load(names, suf): return [np.load(P / f'nn_{m}_{suf}.npy') for m in names]
W = {b: load(FD16, s) for b, s in (('ft', 'n2cpv3s3_val_oof'), ('to', 'trainonly_val'))}
Q = {b: load(FDP, s) for b, s in (('ft', 'n2cpv3s3_val_oof'), ('to', 'trainonly_val'))}
print('rows', len(yv), 'feeds', len(Mv), 'raw member score sd within feed (first member, TO):', float(np.mean([W['to'][0][o:o+m].std() for o, m in zip(ov[:2000], Mv[:2000])])))
res = {}
for name, f in FORMS.items():
    br = {}
    for b in ('ft', 'to'):
        ew = np.mean([f(x) for x in W[b]], 0); ep = np.mean([f(x) for x in Q[b]], 0)
        br[b] = 0.8 * z(ew) + 0.2 * z(ep)      # between-pool and between-branch mixing kept in the file form (z of pooled scores)
    fin = 0.5 * z(br['ft']) + 0.5 * z(br['to'])
    res[name] = ev(fin)
    print(f'{name:12s} final val {res[name].mean():.5f}')
base = res['zmean']
for name in FORMS:
    if name == 'zmean': continue
    d = res[name] - base
    print(f'{name:12s} - zmean: {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f}')
