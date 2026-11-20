"""Mean of seed replicates (per-feed z) for val OOF / test / holdout; full slice report; optional candidate files.
usage: combine.py POOL OUTTAG run1,run2,... [--write]"""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
D = Path('<repo>/nn_x/stack/nnstack/data'); O = Path('<repo>/nn_x/stack/nnstack/out')
P = Path('<repo>/preds'); S = Path('<repo>/subs')
pool, out, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
_, M, o = feed_meta('val'); y = np.load(D / 'val_y.npy'); cold = np.load(D / 'val_cold.npy'); tue = np.load(D / 'val_tue.npy')
fid = np.repeat(np.arange(len(M)), M); npos = np.bincount(fid, y, len(M)); nc = np.bincount(fid, y * cold, len(M))
sl = {'all': np.ones(len(M), bool), 'M<50': M < 50, 'M>=50': M >= 50, 'Tue': tue, 'Thu': ~tue, 'coldpos': (npos > 0) & (nc == npos), 'warmpos': (npos > 0) & (nc == 0)}
z = lambda x, o, M: feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]
base = np.load(D / f'val_pool_{pool}_spec.npy'); nb = ndcg_per_feed(base, y, o, M)
def rep(name, s):
    d = ndcg_per_feed(s, y, o, M) - nb
    print(f'{name:22s} val {ndcg_per_feed(s, y, o, M).mean():.5f} ' + ' '.join(f'{k} {d[m].mean():+.5f}+-{d[m].std() / np.sqrt(m.sum()):.5f}' for k, m in sl.items()), flush=True)
    return d
for r in runs: rep(r, np.load(O / f'{r}_val_oof.npy'))
sv = z(np.mean([z(np.load(O / f'{r}_val_oof.npy'), o, M) for r in runs], 0), o, M)
d = rep('MEAN ' + out, sv)
# fold-level consistency
fold = np.load(D / 'val_fold.npy'); print('per-fold delta', [round(float(d[fold == k].mean()), 5) for k in range(5)])
Mh, oh = np.load(D / 'hold_meta.npy'); yh = np.load(D / 'hold_y.npy'); bh = np.load(D / f'hold_pool_{pool}_to.npy')
hs = [r for r in runs if (O / f'{r}_hold.npy').exists()]
if hs:
    sh = z(np.mean([z(np.load(O / f'{r}_hold.npy'), oh, Mh) for r in hs], 0), oh, Mh)
    dh = ndcg_per_feed(sh, yh, oh, Mh) - ndcg_per_feed(bh, yh, oh, Mh)
    print(f'HOLDOUT OOS ({",".join(hs)}) delta {dh.mean():+.5f} +- {dh.std() / np.sqrt(len(dh)):.5f}')
_, Mt, ot = feed_meta('test')
st = z(np.mean([z(np.load(O / f'{r}_test.npy'), ot, Mt) for r in runs], 0), ot, Mt)
bt = np.load(D / f'test_pool_{pool}_spec.npy')
cv = np.mean([np.corrcoef(sv[o[i]:o[i] + M[i]], base[o[i]:o[i] + M[i]])[0, 1] for i in range(0, len(M), 3) if M[i] > 2])
ct = np.mean([np.corrcoef(st[ot[i]:ot[i] + Mt[i]], bt[ot[i]:ot[i] + Mt[i]])[0, 1] for i in range(0, len(Mt), 3) if Mt[i] > 2])
print(f'per-feed corr vs base: val {cv:.5f} test {ct:.5f}')
np.save(O / f'{out}_val_oof.npy', sv.astype(np.float32)); np.save(O / f'{out}_test.npy', st.astype(np.float32))
if '--write' in sys.argv:
    for n in [P / f'nnstk_{out}_val.npy', P / f'nnstk_{out}_test.npy', S / f'cand_S2_nnstk_{out}.parquet']:
        assert not n.exists(), f'refusing to overwrite {n}'
    from make_sub import make_sub
    np.save(P / f'nnstk_{out}_val.npy', sv.astype(np.float32)); np.save(P / f'nnstk_{out}_test.npy', st.astype(np.float32))
    make_sub(st.astype(np.float32), str(S / f'cand_S2_nnstk_{out}.parquet')); print('written', S / f'cand_S2_nnstk_{out}.parquet')
