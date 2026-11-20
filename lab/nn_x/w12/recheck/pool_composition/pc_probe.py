"""W8P probe at the pre-fixed 0.7 / 0.3 weight (pop1 share as in C3: 8/28 = 0.29), per branch; and the kw4 asymptote; day split."""
from pc_common import *
import pmix
m = Meta(); st = Store(m); G = groups()
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
zv = lambda x: m.z('val', x); zt = lambda x: m.z('test', x); zh = lambda x: m.z('hold', x); big = m.Mv >= 50
day = (m.fv['timestamp'].to_numpy() + 10800) // 86400; print('val days', np.unique(day, return_counts=True))
U = lambda *gs: [v for g in gs for v in G[g]]
W8 = U('w192', 't25w192'); POP = G['pop1']; KD = G['pop1kd']
Wp = pmix.weights()
ftv = lambda ms: np.mean([zv(np.load(P/f'nn_{v}_n2cpv3s3_val_oof.npy')) for v in ms], 0); ftt = lambda ms: np.mean([zt(np.load(P/f'nn_{v}_n2cpv3s3_test.npy')) for v in ms], 0)
wf, wft = ftv(W8), ftt(W8); pf, pft = ftv(POP), ftt(POP)
wt = {sp: st.pool(sp, W8) for sp in ('val', 'test', 'hold')}; pt = {sp: st.pool(sp, POP) for sp in ('val', 'test', 'hold')}; kt = {sp: st.pool(sp, KD) for sp in ('val', 'test', 'hold')}
ref = 0.5 * zv(wf) + 0.5 * zv(wt['val']); reft = 0.5 * zt(wft) + 0.5 * zt(wt['test']); e0 = ev(ref); h0 = eh(wt['hold'])
out = {}
for tag, lam, tov, tot, toh, fv_, ft_ in (('W8P = 0.7 W8 + 0.3 pop1 x8 per branch', 0.7, pt['val'], pt['test'], pt['hold'], pf, pft),
                                         ('W8Pk = FT 0.7 W8 + 0.3 pop1 x8; TO 0.7 W8 + 0.3 pop1kd x4', 0.7, kt['val'], kt['test'], kt['hold'], pf, pft),
                                         ('W8k = FT W8; TO 0.7 W8 + 0.3 pop1kd x4', 0.7, kt['val'], kt['test'], kt['hold'], None, None)):
    f = lam * zv(wf) + (1 - lam) * zv(fv_) if fv_ is not None else zv(wf); ftst = lam * zt(wft) + (1 - lam) * zt(ft_) if ft_ is not None else zt(wft)
    t = lam * zv(wt['val']) + (1 - lam) * zv(tov); tt = lam * zt(wt['test']) + (1 - lam) * zt(tot); th = lam * zh(wt['hold']) + (1 - lam) * zh(toh)
    s = 0.5 * zv(f) + 0.5 * zv(t); stt = 0.5 * zt(ftst) + 0.5 * zt(tt); d = ev(s) - e0; dh = eh(th) - h0
    pm, ps = pmix.wmean_se(d, Wp['private']); qm, qs = pmix.wmean_se(d, Wp['public'])
    days = ' '.join(f'd{int(dd)} {d[day == dd].mean():+.5f}+-{se(d[day == dd]):.5f} (n {int((day == dd).sum())})' for dd in np.unique(day) if (day == dd).sum() > 100)
    print(f'{tag}\n   val {ev(s).mean():.5f} d {fmt(d)} | M>=50 {fmt(d[big])} M<50 {fmt(d[~big])} | {days}\n   pmix private {pm:+.5f}+-{ps:.5f} public {qm:+.5f}+-{qs:.5f} | TO branch val {fmt(ev(t) - ev(wt["val"]))} | TO holdout {fmt(dh)} | test corr to W8 final {np.corrcoef(stt, reft)[0,1]:.4f}')
    out[tag] = dict(val=float(ev(s).mean()), d=float(d.mean()), se=se(d), d_big=float(d[big].mean()), d_big_se=se(d[big]), d_small=float(d[~big].mean()), d_small_se=se(d[~big]),
                    pmix_priv=float(pm), pmix_priv_se=float(ps), pmix_pub=float(qm), pmix_pub_se=float(qs), d_hold=float(dh.mean()), d_hold_se=se(dh), corr_test=float(np.corrcoef(stt, reft)[0, 1]))
kw = U('kw_drop0', 'kw_e20', 'kw_hl2', 'kw_ff4')
for sp, e in (('val', ev), ('hold', eh)):
    A = lambda ms: (len(ms) * e(st.pool(sp, ms)) - np.mean([e(st.get(sp, v)) for v in ms], 0)) / (len(ms) - 1)
    d = A(kw) - A(W8); print(f'asymptote A(kw4 x8) - A(W8) {sp}: {fmt(d)} (A(kw4) {A(kw).mean():.5f})'); out[f'A kw4 {sp}'] = dict(d=float(d.mean()), se=se(d))
json.dump(out, open(OUT/'probe.json', 'w'), indent=1)
