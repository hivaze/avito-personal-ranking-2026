"""Interim run of the pre-declared overnight decision (CURRENT_STATUS 23:45): gates (a)+(b), pools with F6 vs F6*, plus Cd."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
P = Path('<repo>/preds'); H = P / 'holdout'
F6 = ['v13_lr1_mix1','v14_s92','v14_s93','v14_s94','v14_s95','v14_s96']
FAM = {
 'F5': ['v12_all_e10','v12_all_s81','v12_all_s82','v12_allprew_s83','v12_allprew_s84','v12_allprew_s85'],
 'F5cd': ['v15_cd_all_e10','v15_cd_all_s81','v15_cd_all_s82','v15_cd_allprew_s83','v15_cd_allprew_s84','v15_cd_allprew_s85'],
 'F6': F6, 'F6*': [('v21_cd_'+m if m in ('v14_s92','v14_s94','v14_s96') else m) for m in F6],
 'F7': [f'v17_pre2s_s{s}' for s in range(101,108)],
 'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401,407)],
 'F8': [f'v20_pre2s_e2_s{s}' for s in range(301,313)],
}
extra = sys.argv[1:]  # optional extra families "NAME:prefix:seeds-csv"
for e in extra:
    n, pre, ss = e.split(':'); FAM[n] = [f'{pre}{s}' for s in ss.split(',')]
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
print('family: single to val / hold | 6-seed family val mix / hold | gates a+b')
for f in FAM: print(f'  {f:5s} n={len(FAM[f]):2d}  {st[f][0]:.5f} / {st[f][1]:.5f} | {st[f][2]:.5f} / {st[f][3]:.5f} | {"PASS" if ok[f] else "fail"}')
def pool(fs): ms = [m for f in fs for m in FAM[f]]; return mix_v(ms), h(ms)
g6 = [f for f in FAM if ok[f] and f not in ('F6*',)]; g6s = [f for f in FAM if ok[f] and f not in ('F6',)]
C = {'Ca v12 {F5}': (zv(mix_v(FAM['F5'])), h(FAM['F5'])), 'Cd F5cd+F6+F7': pool(['F5cd','F6','F7']),
     'P(F6) '+'+'.join(g6): pool(g6), 'P(F6*) '+'+'.join(g6s): pool(g6s)}
rv, rh = ev(C['Ca v12 {F5}'][0]), eh(C['Ca v12 {F5}'][1])
tue = (fv['timestamp'].to_numpy()+10800)//86400 < 20720
res = {}
for k, (sv, sh) in C.items():
    pv, ph = ev(sv), eh(sh); d = pv - rv; res[k] = ph.mean()
    sl = ' '.join(f'{n} {d[m].mean():+.4f}({d[m].mean()/max(se(d[m]),1e-9):+.1f})' for n, m in [('M<50',Mv<50),('M>=50',Mv>=50),('Tue',tue),('Thu',~tue)])
    print(f'  {k:45s} hold {ph.mean():.5f} | val {pv.mean():.5f} d {d.mean():+.5f}+-{se(d):.5f} z {d.mean()/max(se(d),1e-9):+.2f} | {sl}')
print('holdout-selected:', max((k for k in res if not k.startswith('Ca')), key=res.get))
