"""Hedge / probe candidates that the in-period holdout cannot judge (popularity-light members, the FT branch). Val final, pmix (private-mix
reweighted val; report-only), TO holdout, test corr to the W8 final. No file is written."""
from pc_common import *
import pmix
m = Meta(); st = Store(m); G = groups()
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
zv = lambda x: m.z('val', x); zt = lambda x: m.z('test', x); zh = lambda x: m.z('hold', x); big = m.Mv >= 50
day = (m.fv['timestamp'].to_numpy() + 10800) // 86400; tue = day < 20720
U = lambda *gs: [v for g in gs for v in G[g]]
W8 = U('w192', 't25w192'); POP = G['pop1']; POPKD = G['pop1kd']; C3 = U('F10', 'F7x', 'F7x70', 'pop1')
Wp = pmix.weights(); wpriv = Wp['private']; wpub = Wp['public']
ftv = lambda v: zv(np.load(P/f'nn_{v}_n2cpv3s3_val_oof.npy')); ftt = lambda v: zt(np.load(P/f'nn_{v}_n2cpv3s3_test.npy'))
ftpool = lambda ms: (np.mean([ftv(v) for v in ms], 0), np.mean([ftt(v) for v in ms], 0))
topool = lambda ms: (st.pool('val', ms), st.pool('test', ms), st.pool('hold', ms))
w8f = ftpool(W8); w8t = topool(W8); pf = ftpool(POP); pt = topool(POP); kt = topool(POPKD)
ref_v = 0.5 * zv(w8f[0]) + 0.5 * zv(w8t[0]); ref_t = 0.5 * zt(w8f[1]) + 0.5 * zt(w8t[1]); e0 = ev(ref_v); h0 = eh(w8t[2])
res = {}
def wm(d, w): return pmix.wmean_se(d, w)
def row(tag, fv_, ft_, tv_, tt_, th_, w_ft=0.5):
    sv = w_ft * zv(fv_) + (1 - w_ft) * zv(tv_) if fv_ is not None else zv(tv_)
    stt = w_ft * zt(ft_) + (1 - w_ft) * zt(tt_) if ft_ is not None else zt(tt_)
    d = ev(sv) - e0; dh = eh(th_) - h0; pm, ps = wm(d, wpriv); qm, qs = wm(d, wpub)
    r = dict(val=float(ev(sv).mean()), d=float(d.mean()), se=se(d), d_big=float(d[big].mean()), d_big_se=se(d[big]), d_small=float(d[~big].mean()), d_small_se=se(d[~big]), d_tue=float(d[tue].mean()), d_thu=float(d[~tue].mean()),
             pmix_priv=float(pm), pmix_priv_se=float(ps), pmix_pub=float(qm), pmix_pub_se=float(qs), d_hold=float(dh.mean()), d_hold_se=se(dh), corr_test=float(np.corrcoef(stt, ref_t)[0, 1]))
    res[tag] = r
    print(f'{tag:58s} val {r["val"]:.5f} d {fmt(d)} | M>=50 {d[big].mean():+.5f}+-{se(d[big]):.5f} M<50 {d[~big].mean():+.5f}+-{se(d[~big]):.5f} | Tue {d[tue].mean():+.5f} Thu {d[~tue].mean():+.5f} | pmix priv {pm:+.5f}+-{ps:.5f} pub {qm:+.5f}+-{qs:.5f} | TO hold {fmt(dh)} | test corr {r["corr_test"]:.4f}')
print(f'W8 final val {e0.mean():.5f}, TO holdout {h0.mean():.5f}; pop1 x8 alone: val TO {ev(pt[0]).mean():.5f} FT {ev(pf[0]).mean():.5f} hold {eh(pt[2]).mean():.5f}')
row('W8 TO-only (w_ft 0)', None, None, w8t[0], w8t[1], w8t[2])
row('W8, w_ft 0.3', w8f[0], w8f[1], w8t[0], w8t[1], w8t[2], 0.3)
mix = lambda a, b, lam, z: lam * z(a) + (1 - lam) * z(b)
for lam in (0.5, 0.67, 0.8, 0.9):
    row(f'{lam:.2f} z(W8) + {1-lam:.2f} z(pop1 x8) per branch', mix(w8f[0], pf[0], lam, zv), mix(w8f[1], pf[1], lam, zt), mix(w8t[0], pt[0], lam, zv), mix(w8t[1], pt[1], lam, zt), mix(w8t[2], pt[2], lam, zh))
for lam in (0.67, 0.8):
    row(f'FT W8; TO {lam:.2f} z(W8) + {1-lam:.2f} z(pop1kd x4)', w8f[0], w8f[1], mix(w8t[0], kt[0], lam, zv), mix(w8t[1], kt[1], lam, zt), mix(w8t[2], kt[2], lam, zh))
    row(f'FT {lam:.2f} W8 + {1-lam:.2f} pop1 x8; TO {lam:.2f} W8 + {1-lam:.2f} pop1kd x4', mix(w8f[0], pf[0], lam, zv), mix(w8f[1], pf[1], lam, zt), mix(w8t[0], kt[0], lam, zv), mix(w8t[1], kt[1], lam, zt), mix(w8t[2], kt[2], lam, zh))
c3f = ftpool(C3); c3t = topool(C3)
row('C3 (reference: public -0.00103 vs W8)', c3f[0], c3f[1], c3t[0], c3t[1], c3t[2])
s16 = W8 + U('kd05', 'kd05t25'); sf = ftpool(s16); s_t = topool(s16)
row('S16 (reference: public -0.00041 vs W8)', sf[0], sf[1], s_t[0], s_t[1], s_t[2])
json.dump(res, open(OUT/'hedge.json', 'w'), indent=1)
