"""7) Are user / item embeddings static or snapshot-dependent? (possible time leakage)
 - user_emb of consecutive feeds of the same user (train, and val vs test, train vs val)
 - item embedding vectors of the same item_id across split tables (train vs val vs test)."""
import json
import numpy as np
import polars as pl
from common import feeds, arr

def emb_table(split):
    meta = json.load(open(f'<repo>/data/competition/{split}/item_embeddings/data.meta'))
    return np.memmap(f'<repo>/data/competition/{split}/item_embeddings/data.memmap', dtype=np.int8, mode='r', shape=tuple(meta['shape']))

# --- user embeddings
U = {s: np.asarray(arr(s, 'user_emb')) for s in ['val', 'test']}
ft = feeds('train').filter(pl.col('user_id') != 0).sort(['user_id', 'timestamp'])
u = ft['user_id'].to_numpy(); fi = ft['fidx'].to_numpy(); ts = ft['timestamp'].to_numpy()
same = np.where(u[1:] == u[:-1])[0]
rng = np.random.default_rng(0); same = rng.choice(same, 20000, replace=False)
UE = arr('train', 'user_emb')
a = np.asarray(UE[np.sort(fi[same])]).astype(np.float32)
# need aligned pairs; gather individually
A = np.stack([UE[i] for i in fi[same]]).astype(np.float32); B = np.stack([UE[i] for i in fi[same + 1]]).astype(np.float32)
cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
gap = ts[same + 1] - ts[same]
print('train consecutive same-user feeds: identical user_emb frac', (A == B).all(1).mean(), ' cos quantiles', np.quantile(cos, [0.05, 0.25, 0.5, 0.75, 0.95]))
for lo, hi in [(0, 3600), (3600, 86400), (86400, 7 * 86400), (7 * 86400, 1e9)]:
    m = (gap >= lo) & (gap < hi)
    print(f'  gap [{lo},{hi}): n={m.sum()} identical={(A[m]==B[m]).all(1).mean():.3f} median cos={np.median(cos[m]):.3f}')
# random pairs of different users for reference
r1 = rng.choice(len(ft), 20000); r2 = rng.choice(len(ft), 20000)
C = np.stack([UE[i] for i in fi[r1]]).astype(np.float32); D = np.stack([UE[i] for i in fi[r2]]).astype(np.float32)
print('random different-user pairs median cos', np.median((C * D).sum(1) / (np.linalg.norm(C, axis=1) * np.linalg.norm(D, axis=1) + 1e-9)))
# val vs test same user
fv = feeds('val').filter(pl.col('user_id') != 0); fte = feeds('test').filter(pl.col('user_id') != 0)
j = fv.select(['user_id', 'fidx']).join(fte.select(['user_id', 'fidx']), on='user_id', suffix='_t')
A = U['val'][j['fidx'].to_numpy()].astype(np.float32); B = U['test'][j['fidx_t'].to_numpy()].astype(np.float32)
print('val vs test same user pairs', len(j), 'identical', (A == B).all(1).mean(), 'median cos', np.median((A*B).sum(1)/(np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1)+1e-9)))
# --- item embeddings across split tables
for s1, s2 in [('val', 'test'), ('train', 'val')]:
    i1 = np.asarray(arr(s1, 'cand_item_id')); e1 = np.asarray(arr(s1, 'cand_emb_idx'))
    i2 = np.asarray(arr(s2, 'cand_item_id')); e2 = np.asarray(arr(s2, 'cand_emb_idx'))
    if s1 == 'train':
        i1 = i1[:30_000_000]; e1 = e1[:30_000_000]
    d1 = pl.DataFrame({'item': i1, 'e1': e1}).filter(pl.col('e1') >= 0).unique('item')
    d2 = pl.DataFrame({'item': i2, 'e2': e2}).filter(pl.col('e2') >= 0).unique('item')
    jj = d1.join(d2, on='item')
    if len(jj) > 20000:
        jj = jj.sample(20000, seed=0)
    T1 = emb_table(s1); T2 = emb_table(s2)
    A = np.stack([T1[k] for k in jj['e1'].to_numpy()]).astype(np.float32); B = np.stack([T2[k] for k in jj['e2'].to_numpy()]).astype(np.float32)
    cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
    print(f'item emb {s1} vs {s2}: shared items sampled {len(jj)}, identical {(A==B).all(1).mean():.4f}, cos quantiles {np.quantile(cos,[0.01,0.1,0.5])}')
