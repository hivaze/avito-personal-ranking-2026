"""R4: transfer factor fragility (lb_stats#2) from the per-file table; R5: FT(W8)+TO(S16) vs S16 (ft_redesign#3); R6: N-law out-of-sample (W8 + finished extension seeds)."""
from lo_common import *
import itertools
print('== R4 transfer of offline deltas to public (deltas vs G3; table lb_stats 1.2, re-used as data), weighted by 1/SE_pub^2')
T = {'Cd': (-.00131, -.00060, -.00074, .00062), 'Pall': (-.00136, -.00045, -.00038, .00055), 'PF7x': (-.00058, -.00020, -.00010, .00050),
     'M2': (.00062, -.00018, -.00039, .00043), 'S2': (.00056, None, -.00035, .00035), 'TOonly': (.00003, 0., -.00047, .00050),
     'Q1a': (.00036, -.00052, .00007, .00039), 'Q2a': (.00029, -.00002, .00039, .00039), 'C1': (.00057, -.00073, .00057, .00050),
     'C2': (.00018, 0., -.00007, .00015), 'C3': (.00069, -.00073, .00063, .00051), 'W8': (.00133, .00076, .00167, .00074), 'S16': (.00141, .00049, .00126, .00067)}
def slope(names, col=0):
    x = np.array([T[n][col] for n in names], float); y = np.array([T[n][2] for n in names]); w = 1/np.array([T[n][3] for n in names])**2
    k = (w*x*y).sum()/(w*x*x).sum(); sek = 1/np.sqrt((w*x*x).sum()); return k, sek
allv = list(T); old = ['Cd', 'Pall', 'PF7x', 'M2', 'S2', 'TOonly']; new = ['Q1a', 'Q2a', 'C1', 'C2', 'C3', 'W8', 'S16']
for tag, ns in (('modern 13 (+G3 as origin)', allv), ('11-12 files', old), ('11-13/14 files', new), ('11-13/14 without W8,S16', [n for n in new if n not in ('W8', 'S16')]),
                ('11-13/14 without pop1 files (Q1a,Q2a,C1,C3)', ['C2', 'W8', 'S16']), ('all without W8,S16', [n for n in allv if n not in ('W8', 'S16')])):
    k, s = slope(ns); print(f' val predictive slope through origin, {tag:45s} n={len(ns):2d}: k {k:+.2f} +- {s:.2f} (public noise only; files share one public draw, so SE is optimistic)')
print(' leave-one-out of the modern-13 val slope: ' + ' '.join(f'-{n}:{slope([m for m in allv if m!=n])[0]:+.2f}' for n in allv))
hn = [n for n in allv if T[n][1] is not None]
for tag, ns in (('modern 12 holdout', hn), ('11-13/14 holdout', [n for n in new]), ('holdout without pop1 files', [n for n in hn if n not in ('Q1a', 'Q2a', 'C1', 'C3')])):
    k, s = slope(ns, 1); print(f' holdout predictive slope, {tag:30s} n={len(ns)}: k {k:+.2f} +- {s:.2f}')
# joint fit val + holdout, LOO
def joint(ns):
    X = np.array([[T[n][0], T[n][1]] for n in ns]); y = np.array([T[n][2] for n in ns]); w = 1/np.array([T[n][3] for n in ns])**2
    A = (X*w[:, None]).T@X; b = (X*w[:, None]).T@y; return np.linalg.solve(A, b), np.sqrt(np.diag(np.linalg.inv(A)))
c, s = joint(hn); print(f' joint public = a val + b hold (12 files): a {c[0]:+.2f}+-{s[0]:.2f} b {c[1]:+.2f}+-{s[1]:.2f}')
print('   LOO: ' + ' '.join(f'-{n}:({joint([m for m in hn if m!=n])[0][0]:+.2f},{joint([m for m in hn if m!=n])[0][1]:+.2f})' for n in hn))
print('== R5 ft_redesign#3: S16 file with the FT branch of W8 only')
ft8 = zv(np.load(P/'ens_W8kd_ft_val.npy')); to8 = zv(np.load(P/'ens_W8kd_to_val.npy')); ft16 = zv(np.load(P/'ens_S16kd_ft_val.npy')); to16 = zv(np.load(P/'ens_S16kd_to_val.npy'))
s16 = ev(0.5*ft16+0.5*to16); a = ev(0.5*ft8+0.5*to16); w8 = ev(0.5*ft8+0.5*to8)
print(f' S16 full {s16.mean():.5f}; FT(W8)+TO(S16) {a.mean():.5f}  delta {f(a-s16)}; vs W8 {f(a-w8)}')
t = lambda n: zt(np.load(P/f'ens_{n}_test.npy'))
c = np.corrcoef(0.5*t('W8kd_ft')+0.5*t('S16kd_to'), 0.5*t('S16kd_ft')+0.5*t('S16kd_to'))[0, 1]; print(f' test corr of the two S16 variants {c:.5f}')
print('== R6 N-law out of sample: W8 + finished extension seeds')
ext = [m for m in E('kd05w192', (955, 956, 957, 958))+E('kd05t25w192', (955, 956, 957, 958)) if (W/f'nn_{m}_trainonly_val.npy').exists() and (W/f'holdout/nn_{m}_holdout.npy').exists()]
print(' finished extension members:', ext)
ZV = {m: LV(m) for m in W8+ext}; ZH = {m: LH(m) for m in W8+ext}
w8v, w8h = ev(np.mean([ZV[m] for m in W8], 0)), eh(np.mean([ZH[m] for m in W8], 0))
if ext:
    allm = W8+ext; av, ah = ev(np.mean([ZV[m] for m in allm], 0)), eh(np.mean([ZH[m] for m in allm], 0))
    print(f' W{len(allm)} val {av.mean():.5f} hold {ah.mean():.5f} | minus W8: val {f(av-w8v)} hold {f(ah-w8h)}')
    em = np.mean([ev(ZV[m]).mean() for m in ext]), np.mean([eh(ZH[m]).mean() for m in ext]); om = np.mean([ev(ZV[m]).mean() for m in W8]), np.mean([eh(ZH[m]).mean() for m in W8])
    print(f' member mean: extension {em[0]:.5f}/{em[1]:.5f} vs original 8 {om[0]:.5f}/{om[1]:.5f}')
    rng = np.random.default_rng(3); sub = [rng.choice(len(allm), 8, replace=False) for _ in range(60)]
    sv = [ev(np.mean([ZV[allm[i]] for i in s], 0)).mean() for s in sub]; sh = [eh(np.mean([ZH[allm[i]] for i in s], 0)).mean() for s in sub]
    print(f' random 8-subsets of the {len(allm)}: val {np.mean(sv):.5f} sd {np.std(sv):.5f} | hold {np.mean(sh):.5f} sd {np.std(sh):.5f}  (realised W8 {w8v.mean():.5f}/{w8h.mean():.5f}: val percentile {np.mean(np.array(sv)<w8v.mean()):.2f}, hold {np.mean(np.array(sh)<w8h.mean()):.2f})')
