"""w6: the CatBoost model of record was early-stopped on val user-half A (nn_x/cb26/data/e3_xm_yeti.json). Read level, gap and mixes on half B (clean) and half A separately. Val labels only to score."""
import sys
sys.path.insert(0, '<repo>/nn_x/w12/next/merge_verify/r2')
from w_common import *
V = Split('val'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64); ev = lambda s: my_ndcg(V, s, yv)
hb = np.load(ROOT / 'nn_x/cb26/data/e3_xm_yeti_halfB_feeds.npy'); print('halfB array:', hb.dtype, hb.shape, hb[:5], hb.min(), hb.max())
B = np.zeros(V.n, bool)
if hb.dtype == bool and len(hb) == V.n: B = hb.copy()
elif hb.max() < V.n and len(hb) < V.n: B[hb.astype(np.int64)] = True
else: B = np.isin(V.feed_id, hb)
print(f'half B feeds {int(B.sum())}, half A {int((~B).sum())}')
cb = V.z(np.load(ROOT / 'nn_x/cb26/data/e3_xm_yeti_val.npy')); ecb = ev(cb)
for lab, tag in (('W8 final', 'W8kd'), ('W16 final', 'W16kd')):
    z = V.z(fin(V, tag)); e = ev(z); m5 = ev(0.5 * z + 0.5 * cb) - e; m1 = ev(0.9 * z + 0.1 * cb) - e; rf = V.rf(z, cb)
    for hl, m in (('all', np.ones(V.n, bool)), ('half A (early stopping)', ~B), ('half B (clean)', B)):
        g = (e - ecb)[m]; rho = rf[m].mean()
        print(f'  {lab:9s} {hl:24s} feeds {int(m.sum()):5d}: pool {e[m].mean():.5f} CatBoost {ecb[m].mean():.5f} gap {pm(g)} rho {rho:.4f} | law on record {0.0186 * (1 - rho) - 0.378 * g.mean():+.5f} | equal mix {pm(m5[m])} | w 0.1 {pm(m1[m])}')
