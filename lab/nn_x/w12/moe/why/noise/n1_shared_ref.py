"""why/noise N1: shared-reference confound. Every arm pool (seeds 951-954, selected checkpoints) re-read against
  (a) Ra = reference 951-954 (the screen read), (b) Rb = reference 955-958 (independent seeds), (c) every one of the 70 4-subsets of the
  8 references (distribution), (d) Rbar = per-feed mean NDCG over the 70 4-subsets (a matched-N same-recipe reference).
Also: where Ra and Rb sit in the distribution of the 70 reference 4-pools; decomposition arm - Ra = (arm - Rbar) + (Rbar - Ra).
All deltas paired per feed; SE = per-feed SE (feed sampling only, n = 14,423 val feeds / 30,000 holdout feeds)."""
import itertools, json
from nz_common import *

m = meta()
REF = [nm('kd05w192', s) for s in S8]
subs = list(itertools.combinations(range(8), 4))
pv, ph = {}, {}
for c in subs:
    pv[c], ph[c] = pool_ndcg([REF[i] for i in c])
cA, cB = (0, 1, 2, 3), (4, 5, 6, 7)
Rbar_v = np.mean([pv[c] for c in subs], 0); Rbar_h = np.mean([ph[c] for c in subs], 0)
R8v, R8h = pool_ndcg(REF)
mv = np.array([pv[c].mean() for c in subs]); mh = np.array([ph[c].mean() for c in subs])
out = []
P = lambda *a: (print(*a), out.append(' '.join(str(x) for x in a)))
P('== reference 4-pools (kd05w192, 70 subsets of s951-958)')
P(f'  val     mean {mv.mean():.5f} sd {mv.std(ddof=1):.5f} min {mv.min():.5f} max {mv.max():.5f} | Ra(951-954) {pv[cA].mean():.5f} rank {int((mv < pv[cA].mean()).sum()) + 1}/70'
  f' | Rb(955-958) {pv[cB].mean():.5f} rank {int((mv < pv[cB].mean()).sum()) + 1}/70 | 8-pool {R8v.mean():.5f}')
P(f'  holdout mean {mh.mean():.5f} sd {mh.std(ddof=1):.5f} min {mh.min():.5f} max {mh.max():.5f} | Ra(951-954) {ph[cA].mean():.5f} rank {int((mh < ph[cA].mean()).sum()) + 1}/70'
  f' | Rb(955-958) {ph[cB].mean():.5f} rank {int((mh < ph[cB].mean()).sum()) + 1}/70 | 8-pool {R8h.mean():.5f}')
d = pv[cA] - pv[cB]; e = ph[cA] - ph[cB]
P(f'  Ra - Rb (paired per feed): val {fmt(d)}  holdout {fmt(e)}')
d = pv[cA] - Rbar_v; e = ph[cA] - Rbar_h
P(f'  Ra - Rbar70            : val {fmt(d)}  holdout {fmt(e)}   <- the shared "reference draw" component in every screen delta')
# disjoint splits: 35 (S, S^c) pairs
D35v = np.array([pv[c].mean() - pv[tuple(sorted(set(range(8)) - set(c)))].mean() for c in subs if 0 in c])
D35h = np.array([ph[c].mean() - ph[tuple(sorted(set(range(8)) - set(c)))].mean() for c in subs if 0 in c])
P(f'  35 disjoint 4-vs-4 splits: rms(D) val {np.sqrt((D35v ** 2).mean()):.5f} holdout {np.sqrt((D35h ** 2).mean()):.5f} | Ra-Rb val {D35v[0]:+.5f} '
  f'rank |D| {int((np.abs(D35v) > abs(D35v[0])).sum()) + 1}/35, holdout {D35h[0]:+.5f} | corr(Dval, Dhold) over splits {np.corrcoef(D35v, D35h)[0, 1]:+.2f}')
res = dict(ref_val=mv.tolist(), ref_hold=mh.tolist(), D35v=D35v.tolist(), D35h=D35h.tolist())
P('')
P('== arm pools (951-954) vs the different references (paired per feed)')
for k, a in ARMS.items():
    av, ah = pool_ndcg([nm(a, s) for s in S4A])
    rows = {}
    for lab, rv, rh in (('Ra 951-954', pv[cA], ph[cA]), ('Rb 955-958', pv[cB], ph[cB]), ('Rbar70', Rbar_v, Rbar_h), ('R8 8-pool', R8v, R8h)):
        dv, dh = av - rv, ah - rh; rows[lab] = (dv.mean(), se(dv), dh.mean(), se(dh))
        P(f'  {k} vs {lab:11s}: val {fmt(dv)} (z {dv.mean() / se(dv):+.2f}) | holdout {fmt(dh)} (z {dh.mean() / se(dh):+.2f})')
    dvs = np.array([av.mean() - pv[c].mean() for c in subs]); dhs = np.array([ah.mean() - ph[c].mean() for c in subs])
    P(f'  {k} vs each of 70 ref 4-pools: val median {np.median(dvs):+.5f} [min {dvs.min():+.5f}, max {dvs.max():+.5f}], share < 0 {np.mean(dvs < 0):.2f}'
      f' | holdout median {np.median(dhs):+.5f} [min {dhs.min():+.5f}, max {dhs.max():+.5f}], share > 0 {np.mean(dhs > 0):.2f}'
      f' | share (val<0 & hold>0) {np.mean((dvs < 0) & (dhs > 0)):.2f}')
    P(f'  {k} decomposition: (arm - Ra) val {rows["Ra 951-954"][0]:+.5f} = (arm - Rbar70) {rows["Rbar70"][0]:+.5f} + (Rbar70 - Ra) {Rbar_v.mean() - pv[cA].mean():+.5f};'
      f' holdout {rows["Ra 951-954"][2]:+.5f} = {rows["Rbar70"][2]:+.5f} + {Rbar_h.mean() - ph[cA].mean():+.5f}')
    res[k] = dict(rows={kk: list(vv) for kk, vv in rows.items()}, vs70_val=dvs.tolist(), vs70_hold=dhs.tolist())
P('')
# historical w192 arms: same pattern against Ra?
HIST8 = ['kd05t25w192', 'kd08w192', 'kwwd0', 'kwwd01', 'kwwd05', 'kwaux0', 'kd05w192ps']
HIST4 = ['kd05w224', 'kd05w192gru', 'kdmixw192', 'kwaux1', 'kwd2', 'kwh30', 'kwh6', 'kwr32', 'kwsb2', 'kwwd10', 'kw256d2', 'kd05g2w192', 'kd03g2w192']
P('== historical w192-family arms, 4-pool of seeds 951-954 vs Ra (the same shared reference), and 955-958 vs Rb when present')
P(f'  {"arm":12s} {"val vs Ra":>20s} {"hold vs Ra":>20s} | {"val vs Rbar70":>20s} {"hold vs Rbar70":>20s} | {"val b-half vs Rb":>20s} {"hold b-half vs Rb":>20s}')
H = {}
for a in HIST8 + HIST4:
    av, ah = pool_ndcg([nm(a, s) for s in S4A])
    r = [av - pv[cA], ah - ph[cA], av - Rbar_v, ah - Rbar_h]
    if a in HIST8:
        bv, bh = pool_ndcg([nm(a, s) for s in S4B]); r += [bv - pv[cB], bh - ph[cB]]
    H[a] = [(float(x.mean()), se(x)) for x in r]
    P(f'  {a:12s} ' + ' '.join(f'{x.mean():+.5f}+-{se(x):.5f}' for x in r[:2]) + ' | ' + ' '.join(f'{x.mean():+.5f}+-{se(x):.5f}' for x in r[2:4])
      + (' | ' + ' '.join(f'{x.mean():+.5f}+-{se(x):.5f}' for x in r[4:]) if len(r) > 4 else ''))
va = np.array([H[a][0][0] for a in H]); ha = np.array([H[a][1][0] for a in H])
P(f'  all {len(H)} historical arms vs Ra: mean val delta {va.mean():+.5f} (sd {va.std(ddof=1):.5f}), mean holdout delta {ha.mean():+.5f} (sd {ha.std(ddof=1):.5f});'
  f' share val<0 {np.mean(va < 0):.2f}, share holdout>0 {np.mean(ha > 0):.2f}, share (val<0 & hold>0) {np.mean((va < 0) & (ha > 0)):.2f}; corr(val, hold) {np.corrcoef(va, ha)[0, 1]:+.2f}')
a8 = np.array([H[a][0][0] for a in HIST8]); b8 = np.array([H[a][4][0] for a in HIST8]); a8h = np.array([H[a][1][0] for a in HIST8]); b8h = np.array([H[a][5][0] for a in HIST8])
P(f'  8-seed arms (n={len(HIST8)}): mean (a-half vs Ra) val {a8.mean():+.5f} holdout {a8h.mean():+.5f} | mean (b-half vs Rb) val {b8.mean():+.5f} holdout {b8h.mean():+.5f}'
  f' | paired diff over arms val {np.mean(a8 - b8):+.5f}+-{np.std(a8 - b8, ddof=1) / np.sqrt(len(a8)):.5f} holdout {np.mean(a8h - b8h):+.5f}+-{np.std(a8h - b8h, ddof=1) / np.sqrt(len(a8)):.5f}')
res['hist'] = H
open(OUT / 'n1_shared_ref.txt', 'w').write('\n'.join(out) + '\n')
json.dump(res, open(OUT / 'n1_shared_ref.json', 'w'))
