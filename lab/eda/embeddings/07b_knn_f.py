"""Nonlinear check: kNN (cosine, k=20) regression of item-level mean f1..f5 and item target rate from item embedding (val items, split by item)."""
import numpy as np, polars as pl, faiss
from common import arr, item_table, gather_emb
from emb_features import l2n
faiss.omp_set_num_threads(32)
ci = np.asarray(arr('val', 'cand_item_id')); ce = np.asarray(arr('val', 'cand_emb_idx')); cc = np.asarray(arr('val', 'cand_category_id'))
v = pl.DataFrame({'item': ci, 'idx': ce, 'cat': cc, **{f'f{k}': np.asarray(arr('val', f'cand_f{k}')) for k in range(1, 6)},
                  't': np.asarray(arr('val', 'cand_target')).astype(np.float32)}).filter(pl.col('idx') >= 0)
iv = v.group_by('item').agg(pl.col('idx').first(), pl.col('cat').first(), *[pl.col(f'f{k}').mean() for k in range(1, 6)], pl.col('t').mean(), pl.len().alias('n')).sample(fraction=1.0, seed=3)
X, _ = l2n(gather_emb(item_table('val'), iv['idx'].to_numpy()))
n = len(X); ntr = int(0.8 * n)
import torch
Xg = torch.tensor(X, device='cuda', dtype=torch.float16)
Is = []
for i in range(ntr, n, 8192):
    S = Xg[i:i + 8192] @ Xg[:ntr].T
    Is.append(S.topk(20, dim=1).indices.cpu().numpy())
I = np.concatenate(Is)
cat = iv['cat'].to_numpy()
for c in ['f1', 'f2', 'f3', 'f4', 'f5', 't']:
    y = iv[c].to_numpy().astype(np.float64)
    pred = y[:ntr][I].mean(1)
    # category-mean baseline
    cm = pl.DataFrame({'c': cat[:ntr], 'y': y[:ntr]}).group_by('c').agg(pl.col('y').mean())
    cmap = dict(zip(cm['c'].to_list(), cm['y'].to_list()))
    pc = np.array([cmap.get(k, y[:ntr].mean()) for k in cat[ntr:]])
    yt = y[ntr:]
    r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    r2c = 1 - ((yt - pc) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    print(f'{c}: kNN20 R2 {r2:.3f} | category-mean R2 {r2c:.3f} | corr(kNN) {np.corrcoef(pred, yt)[0, 1]:.3f}')
print('items', n)
