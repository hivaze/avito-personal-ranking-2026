"""M1: what a same-root re-cool is worth as an extra pool member (snapshot / FGE-style ensemble growth), on stored predictions.
CPU only, read-only on every prediction file. Members (all @last, train-only):
  orig = w5_e10_{kd05w192,kd05t25w192}_s951-958 (16 roots, 10-epoch held-out students)
  ref / ref2 = two m02 re-cools of each root's @77500 snapshot with different seeds (w5_{k,l}cd_ref[2]_s951-958)
Witnesses: val (14,423 feeds), in-period holdout (30,000 feeds), stale-val (faithful scenario lt3, nn_x/w12/stale/hz2n_review/preds_stale_lt3).
Run: cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ... .venv/bin/python reports/TRICKS_1115/web_ensembles_meas/m1_recool_growth.py
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
m = Meta()
ST = ROOT / 'nn_x/w12/stale/hz2n_review/preds_stale_lt3'
E = dict(val=lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv), hold=lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh))
E['stale'] = E['val']
def name(kind, st, s):
    if kind == 'orig': return f'w5_e10_{"kd05w192" if st == "k" else "kd05t25w192"}_s{s}'
    return f'w5_{st}cd_{kind}_s{s}'
def load(v, sp):
    if sp == 'stale': return m.z('val', np.load(ST / f'nn_{v}@last_staleval.npy').astype(np.float32))
    pv, ph, _ = paths(v, '@last'); return m.z(sp, np.load(pv if sp == 'val' else ph).astype(np.float32))
roots = [(st, s) for st in 'kl' for s in range(951, 959)]
for sp in ('val', 'hold', 'stale'):
    e = E[sp]
    Z = {k: [load(name(k, st, s), sp) for st, s in roots] for k in ('orig', 'ref', 'ref2')}
    nd = {k: [e(z) for z in Z[k]] for k in Z}
    print(f'\n===== witness {sp} ({len(nd["ref"][0])} feeds) =====')
    for k in Z: print(f'  member mean NDCG {k:5s}: {np.mean([x.mean() for x in nd[k]]):.5f}')
    cz = lambda a, b: float(np.mean(a * b) / np.sqrt(np.mean(a * a) * np.mean(b * b)))
    same = [cz(Z['ref'][i], Z['ref2'][i]) for i in range(16)]
    same_o = [cz(Z['ref'][i], Z['orig'][i]) for i in range(16)] + [cz(Z['ref2'][i], Z['orig'][i]) for i in range(16)]
    diff = [cz(Z['ref'][i], Z['ref2'][j]) for i in range(16) for j in range(16) if i != j]
    diff_o = [cz(Z['orig'][i], Z['orig'][j]) for i in range(16) for j in range(i + 1, 16)]
    print(f'  corr same root  ref x ref2      : {np.mean(same):.4f} (sd {np.std(same):.4f})')
    print(f'  corr same root  re-cool x orig  : {np.mean(same_o):.4f}')
    print(f'  corr diff root  ref x ref2      : {np.mean(diff):.4f}')
    print(f'  corr diff root  orig x orig     : {np.mean(diff_o):.4f}')
    P = lambda zs: e(np.mean(zs, 0))
    p_ref, p_ref2, p_orig = P(Z['ref']), P(Z['ref2']), P(Z['orig'])
    p32, p48 = P(Z['ref'] + Z['ref2']), P(Z['ref'] + Z['ref2'] + Z['orig'])
    print(f'  16-pool ref {p_ref.mean():.5f} | ref2 {p_ref2.mean():.5f} | orig {p_orig.mean():.5f}')
    print(f'  32-pool (ref + ref2, 16 roots x 2 re-cools) {p32.mean():.5f}: minus mean of the two 16-pools {fmt(p32 - 0.5 * (p_ref + p_ref2))}')
    print(f'  48-pool (ref + ref2 + orig)                 {p48.mean():.5f}: minus mean of the three 16-pools {fmt(p48 - (p_ref + p_ref2 + p_orig) / 3)}; minus orig 16-pool {fmt(p48 - p_orig)}')
    # pair level: single, same-root pair, different-root pair (matched: ref_i + ref2_j, j = next root in the same teacher group)
    single = np.mean([0.5 * (nd['ref'][i] + nd['ref2'][i]) for i in range(16)], 0)
    same2 = np.mean([e(0.5 * (Z['ref'][i] + Z['ref2'][i])) for i in range(16)], 0)
    nxt = lambda i: (i // 8) * 8 + (i % 8 + 1) % 8
    diff2 = np.mean([e(0.5 * (Z['ref'][i] + Z['ref2'][nxt(i)])) for i in range(16)], 0)
    g_same, g_diff = same2 - single, diff2 - single
    print(f'  pair level: single {single.mean():.5f} | same-root 2-pool gain {fmt(g_same)} | different-root 2-pool gain {fmt(g_diff)} | ratio {g_same.mean() / g_diff.mean():.3f}')
    # equal N = 16: 8 roots x 2 re-cools vs 16 roots x 1 re-cool (mean over 20 random 8-subsets of roots; reference = mean of ref / ref2 16-pools)
    rng = np.random.default_rng(20261115); acc = np.zeros_like(p_ref); K = 20
    for _ in range(K):
        ix = rng.choice(16, 8, replace=False); acc += P([Z['ref'][i] for i in ix] + [Z['ref2'][i] for i in ix]) / K
    print(f'  equal N=16: (8 roots x 2 re-cools) minus (16 roots x 1 re-cool) {fmt(acc - 0.5 * (p_ref + p_ref2))}')
    acc8 = np.zeros_like(p_ref); rng = np.random.default_rng(20261115)
    for _ in range(K):
        ix = rng.choice(16, 8, replace=False); acc8 += 0.5 * (P([Z['ref'][i] for i in ix]) + P([Z['ref2'][i] for i in ix])) / K
    print(f'  growth from 8 roots: (8 roots x 2 re-cools) minus (8 roots x 1) {fmt(acc - acc8)}; (16 roots x 1) minus (8 roots x 1) {fmt(0.5 * (p_ref + p_ref2) - acc8)}')
