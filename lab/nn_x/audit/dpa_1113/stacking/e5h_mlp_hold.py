"""E5h (reverse direction, declared in the report as an addition after E4/E5): the same tiny residual MLP trained on the
30K-feed HOLDOUT (TO form; user-grouped 5-fold inside the holdout, inner early stopping) -> holdout OOF read; the full-holdout
fit (median ES epochs) applied to val: TO form vs the val TO pool, and the G3 spec 0.5 z(model on FT inputs) + 0.5 z(model on
TO inputs) vs G3. Usage: e5h_mlp_hold.py ARM (B1/B2/B3)."""
from lib import *
import time
import torch, torch.nn as nn
from feats import build
torch.set_num_threads(12)
arm = sys.argv[1]
T0 = time.time()
HO = S('hold'); V = S('val')
hf = np.load(CACHE / 'hold_fold.npy')
X, names, pool = build(HO, arm, 'to')
exec(open(OUT / 'mlp_core.py').read())   # padded, Net, loss_fn, predict, flat, train_on
Xp, Pp, Yp, Mp, idx, msk = padded(HO, X, pool)
nr = len(HO.y)
tag = f'{arm}_hold'
eps = []
for s in (1, 2, 3):
    oof = np.zeros(nr)
    for k in range(5):
        inner = [j for j in range(5) if j != k]; esk = inner[(k + s) % 4]
        net, ep = train_on(HO, X, Xp, Pp, Yp, Mp, idx, msk, (hf != k) & (hf != esk), hf == esk, 1000 * s + k)
        eps.append(ep)
        sc = flat(predict(net, Xp, Pp), idx, msk, nr); te = np.repeat(hf == k, HO.M); oof[te] = sc[te]
        print(f'[{time.time()-T0:.0f}s] seed {s} fold {k} es-epoch {ep} t {float(net.t):.2f}', flush=True)
    np.save(CACHE / f'oofh_mlp_{tag}_s{s}.npy', HO.ev(oof)); np.save(CACHE / f'scrh_mlp_{tag}_s{s}_hold.npy', oof.astype(np.float32))
Efull = int(np.median(eps)); print('median es epochs', Efull, eps, flush=True)
for s in (1, 2, 3):
    net, _ = train_on(HO, X, Xp, Pp, Yp, Mp, idx, msk, np.ones(HO.n, bool), None, 7000 + s, epochs_fixed=Efull)
    for form in ('to', 'ftas'):
        Xs, _, ps = build(V, arm, form)
        Xq, Pq, Yq, Mq, iq, mq = padded(V, Xs, ps)
        sc = flat(predict(net, Xq, Pq), iq, mq, len(V.y))
        np.save(CACHE / f'scrh_mlp_{tag}_s{s}_val_{form}.npy', sc.astype(np.float32))
print(f'[{time.time()-T0:.0f}s] done {tag}', flush=True)
