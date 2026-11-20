"""E6: near-copy statistics (within-feed Spearman, top-1 agreement with G3; val and test, public/private split) for the
E1 variants with val z > 0.9 and for the M-conditional w_ft rule (w_ft 0.3 for M<50, 0.6 for M>=50, the full-val a1 fit)."""
from lib import *
import time, pmix
from scipy.special import ndtri
exec(open(OUT / 'e1_pool.py').read().split("VARS = [")[0].split("T0 = time.time()")[1].replace("log = open(OUT / 'e1_pool.txt', 'w')", "log = open(OUT / 'e6_nearcopy.txt', 'w')"))
for v in ['rawc', 'gscale', 'spow0.5', 'lseT0.5', 'trim10']:
    sv = (0.5 * V.z(agg(v, V, 'ft')) + 0.5 * V.z(agg(v, V, 'to'))).astype(np.float64)
    st = (0.5 * TE.z(agg(v, TE, 'ft')) + 0.5 * TE.z(agg(v, TE, 'to'))).astype(np.float64)
    rv, tv = wf_agree(V.off, V.M, sv, g3.astype(np.float64)); rt, tt = wf_agree(TE.off, TE.M, st, g3t.astype(np.float64))
    pub = TE.public
    pr(f'{v:8s} rho val {rv.mean():.5f} test {rt.mean():.5f} (pub {rt[pub].mean():.5f} priv {rt[~pub].mean():.5f}) | top1 agree val {tv.mean():.4f} test {tt.mean():.4f}')
# M-conditional w_ft
def mcond(sp):
    zf = sp.z(np.asarray(sp.Z('ft')).mean(0).astype(np.float32)); zt = sp.z(np.asarray(sp.Z('to')).mean(0).astype(np.float32))
    w = np.where(np.repeat(sp.M, sp.M) < 50, 0.3, 0.6)
    return (w * zf + (1 - w) * zt).astype(np.float64)
sv = mcond(V); st = mcond(TE)
dv = V.ev(sv) - base_v; mu, se = se_cl(dv, V.cl)
rv, tv = wf_agree(V.off, V.M, sv, g3.astype(np.float64)); rt, tt = wf_agree(TE.off, TE.M, st, g3t.astype(np.float64))
pr(f'Mcond w_ft (0.3 | 0.6) full-val rule, IN-SAMPLE val {mu:+.5f} +- {se:.5f} | rho val {rv.mean():.5f} test {rt.mean():.5f} | top1 val {tv.mean():.4f} test {tt.mean():.4f}')
