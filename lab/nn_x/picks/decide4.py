"""PRE-DECLARED 2026-11-12 12:57 UTC (before any F13g / F13l / F7x70-FT / a4u-m02-FT pool or family number was seen):
the gate-filtered m02 pool decision (follows decide3, CURRENT_STATUS 11:31 "the next pre-declaration will gate-filter").

Families: decide3's (F5cd F6 F7 F9 F10 F12 F7x F7b) + argv extras "NAME:prefix:seeds" (F13g, F13l, F7x70, Fa4u-x, ...).
  A family is ELIGIBLE only with >= 6 members whose ft + to + holdout preds all exist (else printed and skipped).
  F7 (superseded by F7x on the same roots), F7b (control) and F5 (v12) never enter a candidate pool.
Gates (a)+(b) as CURRENT_STATUS §3, against the best eligible family on each of the 4 statistics
  (single train-only val, single holdout, 6-member val mix, 6-member holdout mix; within 0.002 on all four).
Candidates (member-equal z-mean, 0.5 z(ft) + 0.5 z(to); holdout = train-only members):
  PF7x   = F5cd+F6+F7x+F9+F10+F12              (current pick 1, reference)
  Pall   = F5cd+F6+F7+F9+F10+F12                (current pick 2, the no-m02 hedge)
  PF7x+N = PF7x + every gated new (argv) family
  G      = every gated family (excluding F7, F7b)
  G-70   = G without F7x70 (only if F7x70 is in G: it shares the F7 roots with F7x)
  M      = gated m02 families only (F7x, F7x70, F13g, F13l, Fa4u-x, ... whichever are gated)
Decision:
  Pick 1 contender = the VAL-best candidate (val is the test-like split; the holdout is a train-period selector).
    It replaces PF7x iff ALL: val vs PF7x > +2 SE (paired per feed); no val slice (M<50, M>=50, Tue, Thu, coldpos)
    below -2 SE vs PF7x; holdout vs PF7x >= -1 SE (paired); src/pmix private-mix delta vs PF7x >= -1 SE; presub_gate PASS.
  Pick 2 = the HOLDOUT-best candidate other than pick 1 (CURRENT_STATUS §3), EXCEPT that Pall stays pick 2 (no-m02 hedge)
    until a submission containing m02 members has a public score >= 0.3905.
Run when F13g and F13l have 6 complete members each (+ F7x70 FT): output decide4_<HHMM>.txt."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
P = Path('<repo>/preds'); H = P / 'holdout'
F6 = ['v13_lr1_mix1','v14_s92','v14_s93','v14_s94','v14_s95','v14_s96']
FAM = {
 'F5cd': ['v15_cd_all_e10','v15_cd_all_s81','v15_cd_all_s82','v15_cd_allprew_s83','v15_cd_allprew_s84','v15_cd_allprew_s85'],
 'F6': F6, 'F7': [f'v17_pre2s_s{s}' for s in range(101,108)], 'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401,407)],
 'F10': [f'v23_gru_pre2s_s{s}' for s in range(501,507)], 'F12': [f'v26_d192_pre2s_s{s}' for s in range(701,708)],
 'F7x': [f'r6_cd_m02_r{s}' for s in range(101,108)], 'F7b': [f'r6_cd_c0_r{s}' for s in range(101,108)],
}
M02 = {'F7x'}; NEW = []
for e in sys.argv[1:]:
    n, pre, ss = e.split(':'); FAM[n] = [f'{pre}{s}' for s in ss.split(',')]; NEW.append(n); M02.add(n)
have = lambda m: all(p.exists() for p in [P/f'nn_{m}_n2cpv3_val_oof.npy', P/f'nn_{m}_trainonly_val.npy', H/f'nn_{m}_holdout.npy'])
for f in list(FAM):
    ok_m = [m for m in FAM[f] if have(m)]
    if len(ok_m) < len(FAM[f]): print(f'  {f}: {len(FAM[f]) - len(ok_m)} member(s) incomplete: {[m for m in FAM[f] if not have(m)]}')
    FAM[f] = ok_m
    if len(ok_m) < 6: print(f'  {f}: only {len(ok_m)} complete members -> NOT ELIGIBLE'); FAM.pop(f); NEW = [x for x in NEW if x != f]
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
print('family: single to val / hold | 6-member val mix / hold | gates a+b   (best:', ' '.join(f'{x:.5f}' for x in b), ')')
for f in FAM: print(f'  {f:7s} n={len(FAM[f]):2d}  {st[f][0]:.5f} / {st[f][1]:.5f} | {st[f][2]:.5f} / {st[f][3]:.5f} | {"PASS" if ok[f] else "fail"}')
def pool(fs): ms = [m for f in fs for m in FAM[f]]; return mix_v(ms), h(ms), len(ms)
old = ['F5cd','F6','F7','F9','F10','F12']; pf7x = ['F5cd','F6','F7x','F9','F10','F12']
gated = [f for f in FAM if ok[f] and f not in ('F7', 'F7b')]; newg = [f for f in NEW if ok[f]]
C = {'PF7x': pf7x, 'Pall': old}
if newg: C['PF7x+' + '+'.join(newg)] = pf7x + newg
C['G=' + '+'.join(gated)] = gated
if 'F7x70' in gated: C['G-70'] = [f for f in gated if f != 'F7x70']
mg = [f for f in gated if f in M02]
if mg: C['M=' + '+'.join(mg)] = mg
R = {k: pool(fs) for k, fs in C.items()}
pv0, ph0 = ev(R['PF7x'][0]), eh(R['PF7x'][1])
tue = (fv['timestamp'].to_numpy()+10800)//86400 < 20720
cold = np.nan_to_num(np.load('<repo>/nn_x/r6/feat_cache/val/it_hist_all_luo.npy').astype(np.float64), nan=0.0) == 0  # r6_eval definition
fid = np.repeat(np.arange(len(Mv)), Mv)
npos = np.bincount(fid, yv, len(Mv)); ncold = np.bincount(fid, yv*cold, len(Mv)); coldpos = (npos > 0) & (ncold == npos)
W = pmix.weights(); out = {}
for k, (sv, sh, n) in R.items():
    pv, ph = ev(sv), eh(sh); d = pv - pv0; dh = ph - ph0
    sl = [('M<50',Mv<50),('M>=50',Mv>=50),('Tue',tue),('Thu',~tue)] + [('coldpos', coldpos)]
    zs = {nm: d[m].mean()/max(se(d[m]),1e-9) for nm, m in sl}
    pm, pse = pmix.wmean_se(d, W['private'])
    out[k] = dict(val=pv.mean(), hold=ph.mean(), dv=d.mean(), sv=se(d), dh=dh.mean(), sh=se(dh), zs=zs, pm=pm, pse=pse, n=n)
    print(f'  {k:40s} n={n:3d} hold {ph.mean():.5f} (vs PF7x {dh.mean():+.5f}+-{se(dh):.5f}) | val {pv.mean():.5f} (vs PF7x {d.mean():+.5f}+-{se(d):.5f} z {d.mean()/max(se(d),1e-9):+.2f})'
          f' | priv-mix {pm:+.5f}+-{pse:.5f} | ' + ' '.join(f'{nm} {z:+.1f}' for nm, z in zs.items()))
c1 = max(out, key=lambda k: out[k]['val'])
o = out[c1]
rep = c1 != 'PF7x' and o['dv'] > 2*o['sv'] and min(o['zs'].values()) >= -2 and o['dh'] >= -o['sh'] and o['pm'] >= -o['pse']
print(f'pick-1 contender (val-best): {c1} -> ' + ('REPLACES PF7x (pending presub_gate PASS)' if rep else 'PF7x stays pick 1'))
p1 = c1 if rep else 'PF7x'
c2 = max((k for k in out if k != p1), key=lambda k: out[k]['hold'])
print(f'pick-2 by rule: holdout-best other than pick 1 = {c2}; hedge clause: Pall stays pick 2 until an m02 submission has public >= 0.3905')
