"""u20wit: the SHORT witness (W0S) under the val / test domain-token convention ('rl' = no dom in the batch -> model.infer_domain = random-like one-hot),
next to the training token ('tr' = [1, 0, 0] for M < 50). B - A member delta paired per feed, pooled over the seeds, and the 2-pool; CPU vs CPU.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 .venv/bin/python nn_x/w12/next/u20wit/rl_short_read.py
Needs cache/sc/w5_e10_u20{a,b}_s95{1,2}__rl.npy with W0S scored (wit_score.py rl W0S ...). Train labels of the witness rows are read for evaluation only."""
from wit_common import *
from stack_val import feed_rank_z
sm = np.load(CACHE / 'sub_meta.npz'); M = sm['M'].astype(np.int64); off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64); y = np.load(CACHE / 'sub/cand_target.npy').astype(np.float32)
fid = np.repeat(np.arange(len(M)), M); w0s = sm['set_id'] == 0; r = w0s[fid]
E = lambda x: ndcg_feeds(np.nan_to_num(x), y, off, M); Z = lambda x: feed_rank_z(off, M, np.nan_to_num(np.asarray(x, np.float32)))[:, 1]
seeds = (951, 952)
for dom in ('tr', 'rl'):
    X = {(a, s): np.load(CACHE / 'sc' / f'w5_e10_u20{a}_s{s}__{dom}.npy') for a in 'ab' for s in seeds}
    for k, x in X.items(): assert not np.isnan(x[r]).any(), (k, dom)
    e = {k: E(x) for k, x in X.items()}; d = np.mean([e['b', s][w0s] - e['a', s][w0s] for s in seeds], 0)
    pool = {a: E(np.mean([Z(X[a, s]) for s in seeds], 0)) for a in 'ab'}; dp = pool['b'][w0s] - pool['a'][w0s]
    print(f'W0S short (n {int(w0s.sum())}), domain token {dom}: A member {np.mean([e["a", s][w0s].mean() for s in seeds]):.5f} B member {np.mean([e["b", s][w0s].mean() for s in seeds]):.5f} | '
          f'B-A member {fmt(d)} [per seed ' + ' '.join(f'{np.mean(e["b", s][w0s] - e["a", s][w0s]):+.5f}' for s in seeds) + f'] | B-A 2-pool {fmt(dp)}')
    for lo, hi in ((5, 10), (10, 20), (20, 35), (35, 50)):
        mk = (M[w0s] >= lo) & (M[w0s] < hi); print(f'    M {lo}-{hi - 1}: n {int(mk.sum())} B-A member {fmt(d[mk])}')
    if dom == 'tr': dtr = d
print(f'rl minus tr of the B-A member delta, paired per feed: {fmt(d - dtr)}')
print('DONE')
