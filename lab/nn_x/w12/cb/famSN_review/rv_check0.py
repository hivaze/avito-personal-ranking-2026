"""Step 0: own W16 vs infra cache; own NDCG vs a slow per-feed loop on a sample; reference arm."""
import sys; sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review')
from rv_common import *
wv, wh = Wit('val'), Wit('hold')
print('feeds', wv.nf, wh.nf, 'rows', len(wv.y), len(wh.y), 'feeds without positive', int((wv.npos == 0).sum()), int((wh.npos == 0).sum()), 'M<50 val', int((wv.M < 50).sum()), 'min M hold', int(wh.M.min()))
zwv, zwh = w16(wv), w16(wh)
C = ROOT / 'nn_x/w12/cb/infra/cache'
for w, zw, f in ((wv, zwv, 'w16_z_val.npy'), (wh, zwh, 'w16_z_hold.npy')):
    c = np.load(C / f); print(f, 'max abs diff own vs cache', float(np.abs(c - zw).max()), 'level own', w.ndcg(zw).mean(), 'level cache', w.ndcg(c).mean())
np.save(HERE / 'out/w16_own_val.npy', zwv.astype(np.float32)); np.save(HERE / 'out/w16_own_hold.npy', zwh.astype(np.float32))
# slow NDCG on 300 feeds
rng = np.random.default_rng(0)
def slow(w, s, f):
    o, m = w.off[f], w.M[f]; sv = s[o:o + m]; yv = w.y[o:o + m]
    if yv.sum() == 0: return 1.0
    idx = sorted(range(m), key=lambda i: (-sv[i], yv[i]))
    dcg = sum(yv[i] / np.log2(r + 2) for r, i in enumerate(idx)); idcg = sum(1 / np.log2(r + 2) for r in range(int(yv.sum())))
    return dcg / idcg
x = np.load(ROOT / 'nn_x/w12/cb/infra/out/ts_lr020_s0_val.npy'); xr = np.round(x, 1)  # rounded = many ties
fs = rng.choice(wv.nf, 300, replace=False)
a = wv.ndcg(xr)[fs]; b = np.array([slow(wv, xr, f) for f in fs]); print('ndcg fast vs slow (tied scores) max abs diff', np.abs(a - b).max())
o = read(wv, wh, zwv, zwh, x, np.load(ROOT / 'nn_x/w12/cb/infra/out/ts_lr020_s0_hold.npy'), 'ts_lr020_s0'); print(line(o)); print(json.dumps(o))
