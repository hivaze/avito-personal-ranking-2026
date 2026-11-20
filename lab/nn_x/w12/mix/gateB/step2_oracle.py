"""Step 2 ORACLE MAP: IN-SAMPLE, DIAGNOSTIC. Per cell of label-free feed properties: un-optimised contrasts with user-clustered SE and in-sample argmax with a user-bootstrap interval."""
from lib2 import *
import json, sys
comp = sys.argv[1]; y = labels().astype(np.float32); f, M, off = meta('val'); g = groups()
q = pools4('val', None, None, tag=comp); E = tensor(q, y, tag=comp); X = features('val', q); r = np.load(CACHE / 'feat_raw.npz')
day = (r['val__ts'] > np.median(r['val__ts'])).astype(int)   # two val days
ts = r['val__ts']; dd = ((ts + 3 * 3600) // 86400).astype(int); day = (dd != dd.min()).astype(int)
cs = r['val__cold_share']
out = []; P = print
def qcells(x, k, name):
    e = np.unique(np.quantile(x, np.linspace(0, 1, k + 1))); idx = np.clip(np.searchsorted(e, x, side='right') - 1, 0, len(e) - 2)
    return [(f'{name} [{e[c]:.3g}, {e[c + 1]:.3g}{")" if c < len(e) - 2 else "]"}', idx == c) for c in range(len(e) - 1)]
CELLS = {'ALL': [('all feeds', np.ones(len(M), bool))],
         'M decile': [(n.replace('M ', 'M ').replace('lM', 'M'), m) for n, m in qcells(M.astype(float), 10, 'M')],
         'M<50|>=50': [('M < 50', M < 50), ('M >= 50', M >= 50)],
         'cold (luo) quintile': qcells(X[:, 3], 5, 'cold'),
         'pmix cold share': [('cs < 15%', cs <= .15), ('cs 15-30%', (cs > .15) & (cs <= .3)), ('cs 30-45%', (cs > .3) & (cs <= .45)), ('cs >= 45%', cs > .45)],
         'H quintile': qcells(r['val__H'], 5, 'H'),
         'day': [('day 1 (Tue)', day == 0), ('day 2 (Thu)', day == 1)],
         'user0': [('user_id != 0', X[:, 2] == 0), ('user_id == 0', X[:, 2] == 1)],
         'f4m quintile': qcells(X[:, 4], 5, 'f4m'), 'mcs quintile': qcells(X[:, 5], 5, 'mcs'), 'cih quintile': qcells(X[:, 6], 5, 'cih'), 'rho quintile': qcells(X[:, 7], 5, 'rho'),
         'M x cold': [(f'{a} & cold {b} median', ma & mb) for a, ma in (('M < 50', M < 50), ('M >= 50', M >= 50)) for b, mb in (('below', X[:, 3] <= np.median(X[:, 3])), ('above', X[:, 3] > np.median(X[:, 3])))],
         'M x rho': [(f'{a} & rho {b} median', ma & mb) for a, ma in (('M < 50', M < 50), ('M >= 50', M >= 50)) for b, mb in (('below', X[:, 7] <= np.median(X[:, 7])), ('above', X[:, 7] > np.median(X[:, 7])))]}
i2, j5 = IP[0.2], IF[0.5]; rng = np.random.default_rng(7)
def cse(d, gg):
    m = d.mean(); s = np.bincount(gg, d - m); return m, np.sqrt((s ** 2).sum()) / len(d)
def boot_arg(A, gg, B=200):
    """A: feeds x K values; argmax of the mean with a user bootstrap 10-90% interval."""
    ug, inv = np.unique(gg, return_inverse=True); S = np.zeros((len(ug), A.shape[1])); np.add.at(S, inv, A)
    w = rng.multinomial(len(ug), np.ones(len(ug)) / len(ug), size=B).astype(np.float64); arg = (w @ S).argmax(1)
    return int(A.mean(0).argmax()), np.quantile(arg, [.1, .9], method='nearest').astype(int)
P(f'ORACLE MAP ({comp}) -- IN-SAMPLE, DIAGNOSTIC ONLY. Contrasts are un-optimised (SE user-clustered); argmax columns are in-sample optima (optimistic), [10%, 90%] = user bootstrap of the argmax.')
P(f'{"cell":38s} {"n":>6s} | {"pop .4-.2":>17s} {"pop 0-.2":>17s} | {"ft .7-.5":>17s} {"ft .3-.5":>17s} | {"w_pop* [10,90]":>17s} {"gain":>8s} | {"w_ft* [10,90]":>17s} {"gain":>8s} | joint* (gain vs PLAIN / vs MRULE)')
fm = lambda t: f'{t[0]:+.5f}+-{t[1]:.5f}'
mr = np.where(M >= 50, E[:, IP[0.4], j5], E[:, i2, j5])
for var, cells in CELLS.items():
    P(f'-- {var}')
    for name, m in cells:
        e = E[m]; gg = g[m]; pl_ = e[:, i2, j5]
        c1 = cse(e[:, IP[0.4], j5] - pl_, gg); c0 = cse(e[:, IP[0.0], j5] - pl_, gg); f7 = cse(e[:, i2, IF[0.7]] - pl_, gg); f3 = cse(e[:, i2, IF[0.3]] - pl_, gg)
        ap, ipv = boot_arg(e[:, :, j5], gg); af, ifv = boot_arg(e[:, i2, :], gg); mm = e.mean(0); ji, jj = np.unravel_index(mm.argmax(), mm.shape)
        gp = mm[ap, j5] - pl_.mean(); gf = mm[i2, af] - pl_.mean(); gj = mm[ji, jj] - pl_.mean(); gjm = mm[ji, jj] - mr[m].mean()
        P(f'{name:38s} {m.sum():6d} | {fm(c1)} {fm(c0)} | {fm(f7)} {fm(f3)} | {WP[ap]:.2f} [{WP[ipv[0]]:.2f},{WP[ipv[1]]:.2f}] {gp:+.5f} | {WF[af]:.2f} [{WF[ifv[0]]:.2f},{WF[ifv[1]]:.2f}] {gf:+.5f} | ({WP[ji]:.2f},{WF[jj]:.2f}) {gj:+.5f} / {gjm:+.5f}')
        out.append(dict(var=var, cell=name, n=int(m.sum()), pop4=c1, pop0=c0, ft7=f7, ft3=f3, wpop=float(WP[ap]), wpop_lo=float(WP[ipv[0]]), wpop_hi=float(WP[ipv[1]]), gpop=float(gp), wft=float(WF[af]), wft_lo=float(WF[ifv[0]]), wft_hi=float(WF[ifv[1]]), gft=float(gf),
                        joint=(float(WP[ji]), float(WF[jj])), gjoint=float(gj), gjoint_vs_mrule=float(gjm)))
P('\nGLOBAL SURFACE: mean val NDCG minus PLAIN (0.2, 0.5), x 1e5; rows w_pop, columns w_ft')
P('w_pop\\w_ft ' + ' '.join(f'{w:6.2f}' for w in WF))
mm = E.mean(0)
for i, w in enumerate(WP): P(f'{w:10.2f} ' + ' '.join(f'{(mm[i, j] - mm[i2, j5]) * 1e5:+6.0f}' for j in range(len(WF))))
P(f'PLAIN {mm[i2, j5]:.6f} | MRULE {mr.mean():.6f} ({fm(cse(mr - E[:, i2, j5], g))}) | best constant (in-sample) ({WP[np.unravel_index(mm.argmax(), mm.shape)[0]]:.2f}, {WF[np.unravel_index(mm.argmax(), mm.shape)[1]]:.2f}) {mm.max():.6f} ({mm.max() - mm[i2, j5]:+.5f})')
# M-rule family surface (in-sample): w_lo x w_hi at cut 50, w_ft 0.5
P('\nM-RULE FAMILY (in-sample, w_ft 0.5, cut 50): rule minus PLAIN x 1e5; rows w_pop on M < 50, columns w_pop on M >= 50')
lo = E[M < 50][:, :, j5].sum(0); hi = E[M >= 50][:, :, j5].sum(0); T = (lo[:, None] + hi[None, :]) / len(M) - mm[i2, j5]
P('lo\\hi ' + ' '.join(f'{w:5.2f}' for w in WP[:13]))
for i in range(9): P(f'{WP[i]:5.2f} ' + ' '.join(f'{T[i, k] * 1e5:+5.0f}' for k in range(13)))
json.dump(out, open(HERE / f'out_step2_{comp}.json', 'w'), indent=1, default=lambda o: [float(v) for v in o])
