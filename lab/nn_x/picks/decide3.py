"""PRE-DECLARED 2026-11-12 11:00 UTC (before any F7x / F7b fine-tune or pool result was seen): m02 integration decision.
Families: decide2 families + F7x = r6_cd_m02_r101-107 (m02 re-cool of F7 roots), F7b = r6_cd_c0_r101-107 (control re-cool),
optional extra families via argv "NAME:prefix:seeds" (e.g. F7x70, F13g, F13l when ready).
Gates (a)+(b) as CURRENT_STATUS §3 (within 0.002 of the best family on val AND holdout; single TO and 6-seed mix).
Candidate pools (member-equal z-mean of 0.5 z(ft) + 0.5 z(to); holdout = train-only members), chosen on the HOLDOUT:
  Pall           = F5cd+F6+F7+F9+F10+F12                (current pick 2)
  Pall+F7x       = Pall + F7x
  Pall/F7x       = Pall with F7 replaced by F7x
  Pall/F7x+extra = Pall/F7x + every gated extra family (argv)
  F7x            = F7x alone (reference)
F7b enters no candidate (dominated by its own m02 arm by design); it is gated for the record only.
Pick 2 <- holdout-best candidate. Pick 1 (v12) is replaced only if val > 2 SE vs v12, no slice < -2 SE, holdout >= 0.32162,
and presub_gate PASS (CURRENT_STATUS §3)."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
P = Path('<repo>/preds'); H = P / 'holdout'
F6 = ['v13_lr1_mix1','v14_s92','v14_s93','v14_s94','v14_s95','v14_s96']
FAM = {
 'F5cd': ['v15_cd_all_e10','v15_cd_all_s81','v15_cd_all_s82','v15_cd_allprew_s83','v15_cd_allprew_s84','v15_cd_allprew_s85'],
 'F6': F6, 'F7': [f'v17_pre2s_s{s}' for s in range(101,108)], 'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401,407)],
 'F10': [f'v23_gru_pre2s_s{s}' for s in range(501,507)], 'F12': [f'v26_d192_pre2s_s{s}' for s in range(701,708)],
 'F7x': [f'r6_cd_m02_r{s}' for s in range(101,108)], 'F7b': [f'r6_cd_c0_r{s}' for s in range(101,108)],
}
EXTRA = []
for e in sys.argv[1:]:
    n, pre, ss = e.split(':'); FAM[n] = [f'{pre}{s}' for s in ss.split(',')]; EXTRA.append(n)
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT/'val/cand_target.npy')
Mh, oh = np.load(H/'holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(H/'holdout_y.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:,1]; zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:,1]
ev = lambda s: ndcg_per_feed(s, yv, ov, Mv); eh = lambda s: ndcg_per_feed(s, yh, oh, Mh); se = lambda d: d.std()/np.sqrt(len(d))
V, HO = {}, {}
for ms in FAM.values():
    for m in ms:
        if m in HO: continue
        V[(m,'ft')] = zv(np.load(P/f'nn_{m}_n2cpv3_val_oof.npy')); V[(m,'to')] = zv(np.load(P/f'nn_{m}_trainonly_val.npy')); HO[m] = zh(np.load(H/f'nn_{m}_holdout.npy'))
def mix_v(ms): return 0.5*zv(np.mean([V[(m,'ft')] for m in ms],0)) + 0.5*zv(np.mean([V[(m,'to')] for m in ms],0))
def h(ms): return zh(np.mean([HO[m] for m in ms],0))
st = {f: (np.mean([ev(V[(m,'to')]).mean() for m in ms]), np.mean([eh(HO[m]).mean() for m in ms]), ev(mix_v(ms[:6])).mean(), eh(h(ms[:6])).mean()) for f, ms in FAM.items()}
b = [max(s[i] for s in st.values()) for i in range(4)]
ok = {f: all(st[f][i] >= b[i] - 0.002 for i in range(4)) for f in FAM}
print('family: single to val / hold | 6-seed family val mix / hold | gates a+b   (best:', ' '.join(f'{x:.5f}' for x in b), ')')
for f in FAM: print(f'  {f:6s} n={len(FAM[f]):2d}  {st[f][0]:.5f} / {st[f][1]:.5f} | {st[f][2]:.5f} / {st[f][3]:.5f} | {"PASS" if ok[f] else "fail"}')
def pool(fs): ms = [m for f in fs for m in FAM[f]]; return mix_v(ms), h(ms)
base = ['F5cd','F6','F7','F9','F10','F12']; repl = ['F5cd','F6','F7x','F9','F10','F12']
C = {'Ca v12 {F5}': None, 'Pall': pool(base), 'Pall+F7x': pool(base+['F7x']), 'Pall/F7x': pool(repl), 'F7x': pool(['F7x'])}
ex = [f for f in EXTRA if ok[f]]
if ex: C['Pall/F7x+' + '+'.join(ex)] = pool(repl + ex)
F5 = ['v12_all_e10','v12_all_s81','v12_all_s82','v12_allprew_s83','v12_allprew_s84','v12_allprew_s85']
for m in F5: V[(m,'ft')] = zv(np.load(P/f'nn_{m}_n2cpv3_val_oof.npy')); V[(m,'to')] = zv(np.load(P/f'nn_{m}_trainonly_val.npy')); HO[m] = zh(np.load(H/f'nn_{m}_holdout.npy'))
C['Ca v12 {F5}'] = (zv(mix_v(F5)), h(F5))
rv, rh = ev(C['Ca v12 {F5}'][0]), eh(C['Ca v12 {F5}'][1]); pv0 = ev(C['Pall'][0]); ph0 = eh(C['Pall'][1])
tue = (fv['timestamp'].to_numpy()+10800)//86400 < 20720
res = {}
for k, (sv, sh) in C.items():
    pv, ph = ev(sv), eh(sh); d = pv - rv; dp = pv - pv0; dh = ph - ph0; res[k] = ph.mean()
    sl = ' '.join(f'{n} {d[m].mean():+.4f}({d[m].mean()/max(se(d[m]),1e-9):+.1f})' for n, m in [('M<50',Mv<50),('M>=50',Mv>=50),('Tue',tue),('Thu',~tue)])
    print(f'  {k:28s} hold {ph.mean():.5f} (vs Pall {dh.mean():+.5f}+-{se(dh):.5f}) | val {pv.mean():.5f} vs v12 {d.mean():+.5f}+-{se(d):.5f} z {d.mean()/max(se(d),1e-9):+.2f} | vs Pall {dp.mean():+.5f}+-{se(dp):.5f} | {sl}')
print('holdout-selected:', max((k for k in res if not k.startswith('Ca')), key=res.get))
