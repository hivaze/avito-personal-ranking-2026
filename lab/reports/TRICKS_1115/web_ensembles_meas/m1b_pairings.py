"""M1b: consistency check of M1's pool-growth read. For each witness and each pairing of the three same-root families (ref, ref2, orig),
32-pool minus the mean of its two 16-pools, overall and per teacher group (K = kd05w192 roots, L = kd05t25w192 roots; 16-pool vs its two 8-pools).
CPU only, read-only."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
m = Meta(); ST = ROOT / 'nn_x/w12/stale/hz2n_review/preds_stale_lt3'
E = dict(val=lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv), hold=lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)); E['stale'] = E['val']
def name(kind, st, s): return f'w5_e10_{"kd05w192" if st == "k" else "kd05t25w192"}_s{s}' if kind == 'orig' else f'w5_{st}cd_{kind}_s{s}'
def load(v, sp):
    if sp == 'stale': return m.z('val', np.load(ST / f'nn_{v}@last_staleval.npy').astype(np.float32))
    pv, ph, _ = paths(v, '@last'); return m.z(sp, np.load(pv if sp == 'val' else ph).astype(np.float32))
roots = [(st, s) for st in 'kl' for s in range(951, 959)]
for sp in ('val', 'hold', 'stale'):
    e = E[sp]; Z = {k: [load(name(k, st, s), sp) for st, s in roots] for k in ('orig', 'ref', 'ref2')}; P = lambda zs: e(np.mean(zs, 0))
    print(f'\n===== witness {sp} =====')
    for a, b in (('ref', 'ref2'), ('ref', 'orig'), ('ref2', 'orig')):
        for tag, ix in (('all16', range(16)), ('K8', range(8)), ('L8', range(8, 16))):
            pa, pb = P([Z[a][i] for i in ix]), P([Z[b][i] for i in ix]); pab = P([Z[a][i] for i in ix] + [Z[b][i] for i in ix])
            print(f'  {a:4s}+{b:4s} {tag:5s}: pools {pa.mean():.5f} / {pb.mean():.5f} -> union {pab.mean():.5f}; union minus mean {fmt(pab - 0.5 * (pa + pb))}')
    # cross-root control at equal member count: K roots of family a + L roots of family b (16 members, 16 roots) vs K roots of a + K roots of b (16 members, 8 roots)
    for a, b in (('ref', 'ref2'), ('ref', 'orig'), ('ref2', 'orig')):
        d = 0.5 * (P([Z[a][i] for i in range(8)] + [Z[b][i] for i in range(8, 16)]) + P([Z[b][i] for i in range(8)] + [Z[a][i] for i in range(8, 16)])) \
            - 0.5 * (P([Z[a][i] for i in range(8)] + [Z[b][i] for i in range(8)]) + P([Z[a][i] for i in range(8, 16)] + [Z[b][i] for i in range(8, 16)]))
        print(f'  {a:4s}/{b:4s}: 16 members on 16 roots minus 16 members on 8 roots (2 per root) {fmt(d)}')
