"""Final-file level (val only; the holdout is blind to the FT branch): final = w z(FT pool) + (1-w) z(TO pool).
 A. member-level FT OOF vs train-only val, students vs hard-label members (does the hard-label fine-tune help a distilled student?).
 B. branch pools and finals for student / C3 sets, symmetric and asymmetric (FT branch from one set, TO branch from another); test corr to the W8 final.
 C. w_ft curve.  D. kw arms: 2-seed asymptote estimator 2*pool2 - member mean, paired vs kd05w192 same seeds."""
from pc_common import *
m = Meta(); st = Store(m); G = groups()
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
zv = lambda x: m.z('val', x); zt = lambda x: m.z('test', x); big = m.Mv >= 50
day = (m.fv['timestamp'].to_numpy() + 10800) // 86400; tue = day < 20720
U = lambda *gs: [v for g in gs for v in G[g]]
W8 = U('w192', 't25w192'); N8 = U('kd05', 'kd05t25'); C3 = U('F10', 'F7x', 'F7x70', 'pop1'); NEW = U('w192new', 't25w192new')
FTV = {}; FTT = {}
def ft(v):
    if v not in FTV:
        FTV[v] = zv(np.load(P/f'nn_{v}_n2cpv3s3_val_oof.npy')); FTT[v] = zt(np.load(P/f'nn_{v}_n2cpv3s3_test.npy'))
    return FTV[v], FTT[v]
ftpool = lambda ms: (np.mean([ft(v)[0] for v in ms], 0), np.mean([ft(v)[1] for v in ms], 0))
topool = lambda ms: (st.pool('val', ms), st.pool('test', ms))
res = {}
print('== A. member level: FT OOF val vs train-only val (paired per feed; mean over members)')
for nm, ms in (('w192', U('w192')), ('t25w192', U('t25w192')), ('kd05', U('kd05')), ('kd05t25', U('kd05t25')), ('kd08', U('kd08')), ('F7x (hard)', U('F7x')), ('F7x70 (hard)', U('F7x70')), ('F10 (hard)', U('F10')), ('pop1 (hard)', U('pop1'))):
    d = np.mean([ev(ft(v)[0]) - ev(st.get('val', v)) for v in ms], 0)
    c = np.mean([np.corrcoef(ft(v)[0], st.get('val', v))[0, 1] for v in ms])
    ftm = np.mean([ev(ft(v)[0]).mean() for v in ms]); tom = np.mean([ev(st.get('val', v)).mean() for v in ms])
    res[f'A {nm}'] = dict(ft=float(ftm), to=float(tom), d=float(d.mean()), se=se(d), d_big=float(d[big].mean()), d_small=float(d[~big].mean()), corr=float(c))
    print(f'  {nm:14s} n={len(ms)}: TO {tom:.5f} -> FT {ftm:.5f}  delta {fmt(d)}  (M>=50 {d[big].mean():+.5f}+-{se(d[big]):.5f}, M<50 {d[~big].mean():+.5f}+-{se(d[~big]):.5f}; Tue {d[tue].mean():+.5f} Thu {d[~tue].mean():+.5f}) corr(FT,TO) {c:.4f}')

print('\n== B. finals: 0.5 z(FT pool) + 0.5 z(TO pool); delta vs the W8 final (paired); test corr to the W8 final')
w8f = ftpool(W8); w8t = topool(W8)
fin = lambda f, t, w=0.5: (w * zv(f[0]) + (1 - w) * zv(t[0]), w * zt(f[1]) + (1 - w) * zt(t[1]))
ref = fin(w8f, w8t); e0 = ev(ref[0])
chk = np.corrcoef(zt(np.load(P/'ens_W8kd_ft_test.npy')) * 0.5 + 0.5 * zt(np.load(P/'ens_W8kd_to_test.npy')), ref[1])[0, 1]
print(f'  W8 final val {e0.mean():.5f} (FT branch {ev(w8f[0]).mean():.5f}, TO branch {ev(w8t[0]).mean():.5f}); rebuild vs preds/ens_W8kd_* test corr {chk:.6f}; corr(FT pool, TO pool) val {np.corrcoef(w8f[0], w8t[0])[0,1]:.4f}')
SETS = {'W8': W8, 'N8': N8, 'S16': W8 + N8, 'S20': W8 + N8 + U('kd08'), 'C3': C3, 'W8+C3 (36, member-equal)': W8 + C3}
TOS = {'W8': W8, 'W11 (W8 + 3 new)': W8 + NEW, 'W8 + t25w192b3 x2': W8 + U('t25w192b3'), 'W16k (W8 + kw4 x8)': W8 + U('kw_drop0', 'kw_e20', 'kw_hl2', 'kw_ff4'),
       'W12 (W8 + g2k05)': W8 + U('g2k05'), 'W16g (W8 + gen2 x8)': W8 + U('g2k05', 'g2k03'),
       'ALLWIDE38': W8 + NEW + U('g2k05', 'g2k03', 'kx05', 'kx08', 'kw_drop0', 'kw_e20', 'kw_hl2', 'kw_ff4', 't25w192b3', 'kd08w192'), 'S16': W8 + N8, 'C3': C3}
rows = []
def row(tag, f, t, w=0.5):
    s = fin(f, t, w); e = ev(s[0]); d = e - e0
    r = dict(tag=tag, val=float(e.mean()), d=float(d.mean()), se=se(d), d_big=float(d[big].mean()), d_big_se=se(d[big]), d_small=float(d[~big].mean()), d_small_se=se(d[~big]),
             d_tue=float(d[tue].mean()), d_thu=float(d[~tue].mean()), corr_test=float(np.corrcoef(s[1], ref[1])[0, 1]), ft=float(ev(f[0]).mean()), to=float(ev(t[0]).mean()))
    rows.append(r)
    print(f'  {tag:52s} val {e.mean():.5f}  d {fmt(d)} | M>=50 {d[big].mean():+.5f}+-{se(d[big]):.5f} M<50 {d[~big].mean():+.5f}+-{se(d[~big]):.5f} | Tue {d[tue].mean():+.5f} Thu {d[~tue].mean():+.5f} | FT {r["ft"]:.5f} TO {r["to"]:.5f} | test corr {r["corr_test"]:.4f}')
for nm, ms in SETS.items(): row(f'FT {nm} + TO {nm}', ftpool(ms), topool(ms))
print('  -- asymmetric: FT branch = W8 (existing fine-tunes), TO branch = larger train-only pool')
for nm, ms in TOS.items(): row(f'FT W8 + TO {nm}', w8f, topool(ms))
print('  -- asymmetric: FT branch from another set, TO branch = W8')
for nm in ('S16', 'S20', 'C3', 'W8+C3 (36, member-equal)'): row(f'FT {nm} + TO W8', ftpool(SETS[nm]), w8t)
print('  -- per-branch z-mix with C3 (lambda on W8)')
c3f = ftpool(C3); c3t = topool(C3)
for lam in (0.5, 0.7, 0.85):
    f = (lam * zv(w8f[0]) + (1 - lam) * zv(c3f[0]), lam * zt(w8f[1]) + (1 - lam) * zt(c3f[1])); t = (lam * zv(w8t[0]) + (1 - lam) * zv(c3t[0]), lam * zt(w8t[1]) + (1 - lam) * zt(c3t[1]))
    row(f'{lam:.2f} W8 + {1-lam:.2f} C3 per branch', f, t)
    row(f'FT: {lam:.2f} W8 + {1-lam:.2f} C3; TO: W8', f, w8t)
res['B'] = rows
print('\n== C. w_ft curve: final = w z(FT) + (1-w) z(TO)   (val; delta vs w = 0.5 of the same pool)')
res['C'] = {}
for nm in ('W8', 'S16', 'C3'):
    f = ftpool(SETS[nm]); t = topool(SETS[nm]); b = ev(fin(f, t, 0.5)[0]); out = []
    for w in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0):
        e = ev(fin(f, t, w)[0]); d = e - b; out.append(dict(w=w, val=float(e.mean()), d=float(d.mean()), se=se(d), d_big=float(d[big].mean()), d_small=float(d[~big].mean())))
    res['C'][nm] = out
    print(f'  {nm:4s} ' + ' | '.join(f'w {o["w"]:.2f}: {o["val"]:.5f} ({o["d"]:+.5f}+-{o["se"]:.5f}; M>=50 {o["d_big"]:+.5f} M<50 {o["d_small"]:+.5f})' for o in out))
    tt = fin(f, t, 0.0)[1]; print(f'       test corr of the TO-only file to the 0.5/0.5 file: {np.corrcoef(tt, fin(f, t, 0.5)[1])[0,1]:.4f}')

print('\n== D. asymptote estimator from 2 seeds:  A2 = 2 * NDCG(pool of s951+s952) - mean member NDCG; paired vs kd05w192 s951/952 (val | hold)')
def a2(sp, e, ms):
    return 2 * e(st.pool(sp, ms)) - np.mean([e(st.get(sp, v)) for v in ms], 0)
r0 = [f'w5_e10_kd05w192_s{s}' for s in (951, 952)]; av0 = a2('val', ev, r0); ah0 = a2('hold', eh, r0)
p0v = ev(st.pool('val', r0)); p0h = eh(st.pool('hold', r0))
print(f'  reference kd05w192 s951/952: member {np.mean([ev(st.get("val", v)).mean() for v in r0]):.5f}/{np.mean([eh(st.get("hold", v)).mean() for v in r0]):.5f} 2-pool {p0v.mean():.5f}/{p0h.mean():.5f} A2 {av0.mean():.5f}/{ah0.mean():.5f}')
res['D'] = {}
for arm in ('kd05t25w192', 'kw_drop0', 'kw_e20', 'kw_hl2', 'kw_ff4', 'kw_id', 'kd05t25w256', 'kd05t25w192b3', 'kd05g2w192', 'kd03g2w192', 'kx05', 'kx08', 'kd05', 'kd03', 'kd08', 'kd10', 'base'):
    ms = [f'w5_e10_{arm}_s{s}' for s in (951, 952)]
    dmv = np.mean([ev(st.get('val', a)) - ev(st.get('val', b)) for a, b in zip(ms, r0)], 0); dmh = np.mean([eh(st.get('hold', a)) - eh(st.get('hold', b)) for a, b in zip(ms, r0)], 0)
    dpv = ev(st.pool('val', ms)) - p0v; dph = eh(st.pool('hold', ms)) - p0h; dav = a2('val', ev, ms) - av0; dah = a2('hold', eh, ms) - ah0
    cv = np.corrcoef(st.get('val', ms[0]), st.get('val', ms[1]))[0, 1]
    # @last (selection-free on the holdout)
    msL = [v + '@last' for v in ms]; r0L = [v + '@last' for v in r0]
    dahL = a2('hold', eh, msL) - a2('hold', eh, r0L); dphL = eh(st.pool('hold', msL)) - eh(st.pool('hold', r0L))
    res['D'][arm] = dict(member_val=float(dmv.mean()), member_val_se=se(dmv), member_hold=float(dmh.mean()), member_hold_se=se(dmh), pool2_val=float(dpv.mean()), pool2_val_se=se(dpv),
                         pool2_hold=float(dph.mean()), pool2_hold_se=se(dph), a2_val=float(dav.mean()), a2_val_se=se(dav), a2_hold=float(dah.mean()), a2_hold_se=se(dah), corr=float(cv),
                         pool2_hold_last=float(dphL.mean()), pool2_hold_last_se=se(dphL), a2_hold_last=float(dahL.mean()), a2_hold_last_se=se(dahL))
    print(f'  {arm:14s} corr {cv:.4f} | member {fmt(dmv)} / {fmt(dmh)} | 2-pool {fmt(dpv)} / {fmt(dph)} | A2 {fmt(dav)} / {fmt(dah)} | @last hold: 2-pool {fmt(dphL)} A2 {fmt(dahL)}')
print(f'  (corr of the reference pair: {np.corrcoef(st.get("val", r0[0]), st.get("val", r0[1]))[0,1]:.4f})')
json.dump(res, open(OUT/'final.json', 'w'), indent=1)
