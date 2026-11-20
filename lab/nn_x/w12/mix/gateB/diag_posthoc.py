"""POST HOC diagnostics D1-D3 (DECLARATION BLOCK 4). None can produce a candidate."""
from gate import *
import itertools
y = labels().astype(np.float32); f, M, off = meta('val'); g = groups(); n = len(M); big = M >= 50; j5 = IF[0.5]
def cs(d): m, se = cluster_se(d, g); return (float(m), float(se))
fm = lambda t: f'{t[0]:+.5f}+-{t[1]:.5f}'
res = {}
# ---------------- D1
print('=== D1 NULL DISTRIBUTION of the M-rule contrast under same-recipe 4-of-16 subsets (pop := subset, base := the other 12); w_ft = 0.5')
def contrasts(q):
    e2 = ev_w(q, 0.2, 0.5, y); e4 = ev_w(q, 0.4, 0.5, y); e25 = ev_w(q, 0.25, 0.5, y); mr = np.where(big, e4, e2)
    return dict(T=cs(mr - e2), T25=cs(mr - e25), L=float((e4 - e2)[big].mean()), S=float((e4 - e2)[~big].mean()), lvl=float(e2.mean()))
def zstack(names):
    return np.stack([member_z('val', m, 0) for m in names]), np.stack([member_z('val', m, 1) for m in names])
def mk(ZF, ZT, ib, ip):
    return {'Bft': Z('val', ZF[ib].mean(0)), 'Bto': Z('val', ZT[ib].mean(0)), 'Pft': Z('val', ZF[ip].mean(0)), 'Pto': Z('val', ZT[ip].mean(0))}
ZS = {}; null = {}
for S in ('FD16', 'W16'):
    ZF, ZT = zstack(MEM[S]); ZS[S] = (ZF, ZT); rng = np.random.default_rng(2026); subs = [tuple(range(0, 4)), tuple(range(4, 8)), tuple(range(8, 12)), tuple(range(12, 16))]
    while len(subs) < 44:
        c = tuple(sorted(rng.choice(16, 4, replace=False).tolist()))
        if c not in subs: subs.append(c)
    rows = []
    for c in subs:
        ib = np.array([i for i in range(16) if i not in c]); r = contrasts(mk(ZF, ZT, ib, np.array(c))); r['subset'] = c; rows.append(r)
    null[S] = rows; T = np.array([r['T'][0] for r in rows]); T25 = np.array([r['T25'][0] for r in rows]); L = np.array([r['L'] for r in rows]); Sh = np.array([r['S'] for r in rows]); se = np.mean([r['T'][1] for r in rows])
    print(f'{S}: 44 subsets | T = M-rule - const 0.2: mean {T.mean():+.5f} sd {T.std(ddof=1):.5f} min {T.min():+.5f} q10 {np.quantile(T, .1):+.5f} q50 {np.median(T):+.5f} q90 {np.quantile(T, .9):+.5f} max {T.max():+.5f} | mean paired SE of one T {se:.5f} | share T > 0: {np.mean(T > 0):.2f}')
    print(f'      T25 = M-rule - const 0.25: mean {T25.mean():+.5f} sd {T25.std(ddof=1):.5f} q10 {np.quantile(T25, .1):+.5f} q90 {np.quantile(T25, .9):+.5f} share > 0 {np.mean(T25 > 0):.2f} | long feeds E(.4)-E(.2): mean {L.mean():+.5f} sd {L.std(ddof=1):.5f} | short feeds E(.4)-E(.2): mean {Sh.mean():+.5f} sd {Sh.std(ddof=1):.5f}')
    print('      structured subsets (recipe x seed half) T:', ' '.join(f'{r["T"][0]:+.5f}' for r in rows[:4]), '| T25:', ' '.join(f'{r["T25"][0]:+.5f}' for r in rows[:4]))
res['D1_null'] = {S: [dict(subset=list(r['subset']), T=r['T'], T25=r['T25'], L=r['L'], S=r['S']) for r in null[S]] for S in null}
PZ = {k: zstack(MEM[k]) for k in ('FDP', 'P1w', 'P1d')}; res['D1_real'] = {}
print('real compositions (base + 4 popularity-light students): T = M-rule - plain 0.2 | long-feed E(.4)-E(.2) | short-feed E(.4)-E(.2) | rank in the null of its own base set | (T - null mean) / null sd')
for S, Pn in (('FD16', 'FDP'), ('W16', 'P1w'), ('W16', 'P1d'), ('FD16', 'P1w'), ('W16', 'FDP'), ('FD16', 'P1d')):
    ZF, ZT = ZS[S]; q = {'Bft': Z('val', ZF.mean(0)), 'Bto': Z('val', ZT.mean(0)), 'Pft': Z('val', PZ[Pn][0].mean(0)), 'Pto': Z('val', PZ[Pn][1].mean(0))}; r = contrasts(q)
    T = np.array([x['T'][0] for x in null[S]]); L = np.array([x['L'] for x in null[S]]); Sh = np.array([x['S'] for x in null[S]])
    zc = (r['T'][0] - T.mean()) / T.std(ddof=1); res['D1_real'][f'{S}+{Pn}'] = dict(T=r['T'], L=r['L'], S=r['S'], rank=int((T < r['T'][0]).sum()), z_null=float(zc), zL=float((r['L'] - L.mean()) / L.std(ddof=1)), zS=float((r['S'] - Sh.mean()) / Sh.std(ddof=1)), lvl=r['lvl'])
    print(f'  {S:5s} + {Pn:4s}: T {fm(r["T"])} | long {r["L"]:+.5f} (null z {(r["L"] - L.mean()) / L.std(ddof=1):+.2f}) | short {r["S"]:+.5f} (null z {(r["S"] - Sh.mean()) / Sh.std(ddof=1):+.2f}) | above {int((T < r["T"][0]).sum())} of 44 | null z {zc:+.2f} | plain level {r["lvl"]:.5f}')
# ---------------- D2
print('\n=== D2 CROSS-MEMBER-SET NESTED (fold fits of A applied to the held-out users of B; grid families)')
C = {}
for comp in ('FD16P4', 'W16P4'):
    q = pools4('val', None, None, tag=comp); C[comp] = dict(E=np.load(CACHE / f'E_{comp}.npy'), X=features('val', q), J=json.load(open(HERE / f'out_step3_{comp}.json')))
def weights(k, d, X):
    if k in ('CONSTp', 'CONST2'): return np.full(n, d['wp']), np.full(n, d['wf'])
    if k == 'MSTEP': hi = X[:, 0] >= np.log(d['cut']) - 1e-9; return np.where(hi, d['hi'], d['lo']), np.full(n, 0.5)
    if k == 'STUMP1': m = X[:, FEATS.index(d['feat'])] > d['cut']; return np.where(m, d['hi'][0], d['lo'][0]), np.where(m, d['hi'][1], d['lo'][1])
    if k == 'PSTUMP':
        mp = X[:, FEATS.index(d['pop']['feat'])] > d['pop']['cut'] if d['pop']['feat'] else np.zeros(n, bool); mf = X[:, FEATS.index(d['ft']['feat'])] > d['ft']['cut'] if d['ft']['feat'] else np.zeros(n, bool)
        return np.where(mp, d['pop']['hi'], d['pop']['lo']), np.where(mf, d['ft']['hi'], d['ft']['lo'])
res['D2'] = {}
for A, B in (('FD16P4', 'W16P4'), ('W16P4', 'FD16P4')):
    EB = C[B]['E']; XB = C[B]['X']; pl_ = EB[:, IP[0.2], j5]; mr = np.where(big, EB[:, IP[0.4], j5], pl_); ar = np.arange(n)
    print(f'fit on {A} -> scored on {B} (PLAIN {pl_.mean():.6f}, MRULE {mr.mean():.6f}): cross-nested minus PLAIN | minus MRULE | [own-nested of {B} minus PLAIN for comparison]')
    for k in ('CONSTp', 'CONST2', 'MSTEP', 'STUMP1', 'PSTUMP'):
        acc = np.zeros(n); t = 0
        for s in range(5):
            fo = folds(g, s, 5)
            for c in range(5):
                wp, wf = weights(k, C[A]['J']['fam'][k]['folds'][t], XB); t += 1; te = fo == c
                acc[te] += EB[ar[te], np.rint(wp[te] / 0.05).astype(int), np.rint((wf[te] - 0.2) / 0.05).astype(int)] / 5
        res['D2'][f'{A}->{B}:{k}'] = dict(vs_plain=cs(acc - pl_), vs_mrule=cs(acc - mr))
        print(f'  {k:7s} {fm(cs(acc - pl_))} | {fm(cs(acc - mr))} | [{C[B]["J"]["fam"][k]["n_vs_plain"][0]:+.5f}]')
# ---------------- D3
print('\n=== D3 POOLED-TENSOR NESTED (fit on the mean tensor of FD16P4 and W16P4, training users; scored on each composition, held-out users)')
Ep = (C['FD16P4']['E'] + C['W16P4']['E']) / 2; ctx = Ctx(Ep, C['FD16P4']['X'], None, y); acc = {(k, B): np.zeros(n) for k in ('CONSTp', 'MSTEP', 'STUMP1') for B in C}; picks = {k: [] for k in ('CONSTp', 'MSTEP', 'STUMP1')}; ar = np.arange(n)
for s in range(5):
    fo = folds(g, s, 5)
    for c in range(5):
        tr = ar[fo != c]; te = fo == c
        for k, fn in (('CONSTp', ctx.constp), ('MSTEP', ctx.mstep), ('STUMP1', ctx.stump1)):
            wp, wf, d = fn(tr); picks[k].append(d)
            for B in C: acc[(k, B)][te] += C[B]['E'][ar[te], np.rint(wp[te] / 0.05).astype(int), np.rint((wf[te] - 0.2) / 0.05).astype(int)] / 5
from collections import Counter
res['D3'] = {}
for B in C:
    EB = C[B]['E']; pl_ = EB[:, IP[0.2], j5]; mr = np.where(big, EB[:, IP[0.4], j5], pl_)
    for k in ('CONSTp', 'MSTEP', 'STUMP1'):
        res['D3'][f'{k}@{B}'] = dict(vs_plain=cs(acc[(k, B)] - pl_), vs_mrule=cs(acc[(k, B)] - mr)); print(f'  pooled-fit {k:7s} scored on {B}: minus PLAIN {fm(cs(acc[(k, B)] - pl_))} | minus MRULE {fm(cs(acc[(k, B)] - mr))}')
for k in ('CONSTp', 'MSTEP', 'STUMP1'):
    d = (acc[(k, 'FD16P4')] + acc[(k, 'W16P4')]) / 2; plm = Ep[:, IP[0.2], j5]; mrm = np.where(big, Ep[:, IP[0.4], j5], plm)
    res['D3'][f'{k}@mean'] = dict(vs_plain=cs(d - plm), vs_mrule=cs(d - mrm)); print(f'  pooled-fit {k:7s} mean of the two compositions: minus PLAIN {fm(cs(d - plm))} | minus MRULE {fm(cs(d - mrm))}')
print('  MSTEP picks:', Counter((d['cut'], d['lo'], d['hi']) for d in picks['MSTEP']).most_common(6)); print('  CONSTp picks:', Counter(d['wp'] for d in picks['CONSTp']).most_common(5)); print('  STUMP1 picks:', Counter(d['feat'] for d in picks['STUMP1']).most_common(5))
allf = ar; print('  all-feed pooled fits (IN-SAMPLE): MSTEP', ctx.mstep(allf)[2], '| CONSTp', ctx.constp(allf)[2], '| STUMP1', ctx.stump1(allf)[2])
plm = Ep[:, IP[0.2], j5]; mrm = np.where(big, Ep[:, IP[0.4], j5], plm); print(f'  pooled tensor (IN-SAMPLE): PLAIN {plm.mean():.6f} | MRULE - PLAIN {fm(cs(mrm - plm))} | const 0.4 - PLAIN {fm(cs(Ep[:, IP[0.4], j5] - plm))} | const 0.3 - PLAIN {fm(cs(Ep[:, IP[0.3], j5] - plm))}')
print('  pooled M-RULE FAMILY table (IN-SAMPLE, cut 50, w_ft 0.5): rule minus PLAIN x 1e5; rows w_pop on M < 50, columns w_pop on M >= 50')
lo = Ep[~big][:, :, j5].sum(0); hi = Ep[big][:, :, j5].sum(0); T = (lo[:, None] + hi[None, :]) / n - plm.mean()
print('  lo\\hi ' + ' '.join(f'{w:5.2f}' for w in WP[:15]))
for i in range(11): print(f'  {WP[i]:5.2f} ' + ' '.join(f'{T[i, k] * 1e5:+5.0f}' for k in range(15)))
print('  pooled dose by M decile (IN-SAMPLE): argmax w_pop at w_ft 0.5 and E(w) - E(0.2) x 1e5 at w = 0, .1, .3, .4, .5, .6')
e = np.unique(np.quantile(M, np.linspace(0, 1, 11))); idx = np.clip(np.searchsorted(e, M, side='right') - 1, 0, 9)
for c in range(10):
    m = idx == c; mm = Ep[m][:, :, j5].mean(0); print(f'   M [{e[c]:.0f}, {e[c + 1]:.0f}{")" if c < 9 else "]"} n {m.sum()}: argmax {WP[mm.argmax()]:.2f} | ' + ' '.join(f'{(mm[IP[w]] - mm[IP[0.2]]) * 1e5:+5.0f}+-{cluster_se(Ep[m][:, IP[w], j5] - Ep[m][:, IP[0.2], j5], g[m])[1] * 1e5:3.0f}' for w in (0.0, 0.1, 0.3, 0.4, 0.5, 0.6)))
json.dump(res, open(HERE / 'out_posthoc.json', 'w'), indent=1, default=float)
