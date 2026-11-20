"""(a) compare re-inferred predictions (nn path vs r6 path, both vs saved preds/nn_<v>_trainonly_{val,test}.npy)."""
import sys
import numpy as np
sys.path.insert(0, '<repo>/src')
from common import feed_meta
from stack_val import feed_rank_z
from metrics import ndcg_per_feed
R = '<repo>'; O = f'{R}/nn_x/pubdiag/bughunt/out'
yv = np.load(f'{R}/data/flat/val/cand_target.npy').astype(np.float64)
meta = {sp: feed_meta(sp) for sp in ('val', 'test')}
for v in ['v17_pre2s_s101', 'r6_cd_m02_r101', 'r6_cd_m02_p70k_r101', 'r6_cd_c0_r101']:
    for sp in ('val', 'test'):
        _, M, off = meta[sp]
        a = np.load(f'{O}/nn__{v}_{sp}.npy'); b = np.load(f'{O}/r6__{v}_{sp}.npy'); s = np.load(f'{R}/preds/nn_{v}_trainonly_{sp}.npy')
        def cmp(x, y):
            d = np.abs(x.astype(np.float64) - y)
            za, zb = feed_rank_z(off, M, x)[:, 1], feed_rank_z(off, M, y)[:, 1]
            return f'bitexact {np.array_equal(x, y)} maxabs {d.max():.2e} n_diff {(d > 0).sum()} max|dz| {np.abs(za - zb).max():.2e}'
        line = f'{v:22s} {sp:4s} nn-vs-r6: {cmp(a, b)} | r6-vs-saved: {cmp(b, s)} | nn-vs-saved: {cmp(a, s)}'
        if sp == 'val':
            nd = {k: ndcg_per_feed(x, yv, off, M) for k, x in (('nn', a), ('r6', b), ('saved', s))}
            line += f" | val ndcg nn {nd['nn'].mean():.5f} r6 {nd['r6'].mean():.5f} saved {nd['saved'].mean():.5f}"
        print(line)
