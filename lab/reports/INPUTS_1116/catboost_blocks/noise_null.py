"""NULL of the block read of blocks_read.py: the same pipeline on blocks that cannot carry information.
Null used: Gaussian noise columns (z-scored inside each feed like a real column), blocks of 1, 5 and 25 columns.
Output: the distribution of NATS (cross-fitted on user halves, as in blocks_read.py) over the draws -> out/noise_null.txt"""
from cbi_common import *
import json, time
T0 = time.time()
meta = Meta(); WS = {'val': Wit(meta, 'val'), 'hold': Wit(meta, 'hold')}
rng = np.random.default_rng(20261116)
lines = []
def say(s=''):
    print(s, flush=True); lines.append(s)

def read_block(X, w):
    half = w.halves(0); d = np.zeros(w.nf); mx = np.zeros(w.nf)
    for h in (True, False):
        fit = half == h; b, _ = clogit(np.column_stack([w.zw16, X]), w, fit, ridge=1e-6); comp = X @ b[1:]
        dd, b2 = nats(comp, w.zw16, w, ~fit); d[~fit] = dd
        mix = w.ndcg(b2[0] * w.zw16 + b2[1] * comp) - w.ndcg(w.zw16); mx[~fit] = mix[~fit]
    return float(d.mean()), float(mx.mean())

res = {}
for sp, w in WS.items():
    for p, ndraw in ((1, 40), (5, 30), (25, 10)):
        out = []
        for _ in range(ndraw):
            X = np.column_stack([w.zcol(rng.standard_normal(w.n)) for _ in range(p)]); out.append(read_block(X, w))
        a = np.array(out); res[f'{sp}_gauss_p{p}'] = a.tolist()
        say(f'[{sp}] Gaussian noise, {p:2d} columns, {ndraw} draws: NATS mean {a[:, 0].mean():+.6f}, sd {a[:, 0].std():.6f}, max {a[:, 0].max():+.6f}, 95% {np.quantile(a[:, 0], 0.95):+.6f} | dNDCG mean {a[:, 1].mean():+.6f}, sd {a[:, 1].std():.6f}, max {a[:, 1].max():+.6f}')
        json.dump(res, open(CBI_OUT + '/noise_null.json', 'w'))
say(f'analytic check: 1 / (2 n_feeds) = {1 / (2 * WS["val"].nf):.6f} (val), {1 / (2 * WS["hold"].nf):.6f} (holdout)')
open(CBI_OUT + '/noise_null.txt', 'w').write('\n'.join(lines) + f'\n# elapsed {time.time() - T0:.0f} s\n')
