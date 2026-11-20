"""Summary table over all finished stackers + combos + shrink curve + val/test drift diagnostics."""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
D = Path('<repo>/nn_x/stack/nnstack/data'); O = Path('<repo>/nn_x/stack/nnstack/out')
_, M, o = feed_meta('val'); y = np.load(D / 'val_y.npy'); cold = np.load(D / 'val_cold.npy'); tue = np.load(D / 'val_tue.npy')
_, Mt, ot = feed_meta('test'); ct = np.load(D / 'test_cold.npy')
fid = np.repeat(np.arange(len(M)), M); npos = np.bincount(fid, y, len(M)); nc = np.bincount(fid, y * cold, len(M))
sl = {'all': np.ones(len(M), bool), 'M<50': M < 50, 'M>=50': M >= 50, 'Tue': tue, 'Thu': ~tue, 'coldpos': (npos > 0) & (nc == npos), 'warmpos': (npos > 0) & (nc == 0)}
z = lambda x, o=o, M=M: feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]
pool = sys.argv[1] if len(sys.argv) > 1 else 'PF7x'
base = np.load(D / f'val_pool_{pool}_spec.npy'); nb = ndcg_per_feed(base, y, o, M)
baset = np.load(D / f'test_pool_{pool}_spec.npy')
def rep(name, s):
    d = ndcg_per_feed(s, y, o, M) - nb
    print(f'{name:28s} ' + ' '.join(f'{k} {d[m].mean():+.5f}+-{d[m].std() / np.sqrt(m.sum()):.5f}' for k, m in sl.items()), flush=True)
    return d
runs = {}
for f in sorted(O.glob('*_val_oof.npy')):
    t = f.name.replace('_val_oof.npy', '')
    if (('g3' in t) != (pool == 'G3')): continue
    runs[t] = f
for f in sorted(O.glob('h_*_val_both.npy')):
    if ('g3' in f.name) == (pool == 'G3'): runs[f.name.replace('_val_both.npy', '')] = f
ds = {}
for t, f in runs.items(): ds[t] = rep(t, np.load(f))
ks = list(ds); C = np.corrcoef(np.stack([ds[k] for k in ks]))
print('corr of per-feed deltas:'); print('\n'.join(f'{k:20s} ' + ' '.join(f'{c:.2f}' for c in C[i]) for i, k in enumerate(ks)))
def avg(names): return z(np.mean([z(np.load(runs[n])) for n in names], 0))
if pool == 'PF7x':
    for combo in [['v_ds_b3', 'h_ds'], ['v_ds_b3', 's2_ds_pre'], [k for k in ks if k.startswith('v_')], ks]:
        combo = [c for c in combo if c in runs]
        if len(combo) > 1: rep('AVG ' + '+'.join(combo)[:24], avg(combo))
    if 'v_ds_b3' in runs:
        s = z(np.load(runs['v_ds_b3']))
        for w in (0.25, 0.5, 0.75): rep(f'shrink v_ds_b3 w={w}', w * s + (1 - w) * base)
# val vs test drift: mean z-shift of cold items (new - base), val OOF vs test
for t in ks:
    tf = O / f'{t}_test.npy'
    if not tf.exists(): continue
    sv = z(np.load(runs[t])); st = feed_rank_z(ot, Mt, np.load(tf).astype(np.float32))[:, 1]
    print(f'{t:20s} cold-item mean shift val {np.mean((sv - base)[cold > 0]):+.4f} test {np.mean((st - baset)[ct > 0]):+.4f} | '
          f'warm val {np.mean((sv - base)[cold == 0]):+.4f} test {np.mean((st - baset)[ct == 0]):+.4f} | rms shift val {np.sqrt(np.mean((sv - base) ** 2)):.4f} test {np.sqrt(np.mean((st - baset) ** 2)):.4f}')
