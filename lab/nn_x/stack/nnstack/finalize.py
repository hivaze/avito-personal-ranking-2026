"""Tie-safe final arrays: within-feed rank of the stacker score, exact ties broken by the float64 reference pool score
(0.5*z(ft)+0.5*z(to)), saved as float32 integer ranks so the submission and the gate's --test spec share one order."""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from make_sub import make_sub
O = Path('<repo>/nn_x/stack/nnstack/out'); P = Path('<repo>/preds'); S = Path('<repo>/subs')
D = Path('<repo>/nn_x/stack/nnstack/data')
src, pool, out = sys.argv[1], sys.argv[2], sys.argv[3]
for n in [P / f'nnstk_{out}_val.npy', P / f'nnstk_{out}_test.npy', S / f'cand_S2_nnstk_{out}.parquet']:
    assert not n.exists(), f'refusing to overwrite {n}'
res = {}
for split, fn in [('val', f'{src}_val_oof.npy'), ('test', f'{src}_test.npy')]:
    f, M, o = feed_meta(split); fid = np.repeat(np.arange(len(M)), M)
    s = np.load(O / fn).astype(np.float64)
    ref = 0.5 * feed_rank_z(o, M, np.load(P / f'ens_{pool}_ft_{split}.npy').astype(np.float32))[:, 1].astype(np.float64) \
        + 0.5 * feed_rank_z(o, M, np.load(P / f'ens_{pool}_to_{split}.npy').astype(np.float32))[:, 1].astype(np.float64)
    order = np.lexsort((ref, s, fid)); r = np.empty(len(s)); r[order] = np.arange(len(s)) - np.repeat(o, M)
    r = r.astype(np.float32)
    assert all(len(np.unique(r[o[i]:o[i] + M[i]])) == M[i] for i in range(len(M)))
    np.save(P / f'nnstk_{out}_{split}.npy', r); res[split] = (r, ref, M, o)
r, ref, M, o = res['val']; y = np.load(D / 'val_y.npy')
d = ndcg_per_feed(r, y, o, M) - ndcg_per_feed(ref, y, o, M)
print(f'val rank-version delta vs ref {d.mean():+.6f} +- {d.std() / np.sqrt(len(d)):.6f} (raw ref ndcg {ndcg_per_feed(ref, y, o, M).mean():.5f})')
make_sub(res['test'][0], str(S / f'cand_S2_nnstk_{out}.parquet')); print('written', S / f'cand_S2_nnstk_{out}.parquet')
