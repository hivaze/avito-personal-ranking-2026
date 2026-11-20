"""CRITIC check 3 (CPU, read-only): re-derivation of two reads the plan rests on, from the prediction files their authors left on disk, with own code.
A  domain token (synthesis 3.1 item 1): file-level and pool-level deltas, CPU reference.
B  lag-equalised inputs (web_shift 2.5, plan 3.1 item 3): the 4-pool read, and what it becomes at FILE level in a 4 + 4 file of the same members
   (power of the plan's step-2 clause "pmix-private >= +1 SE"). Declared before the run: one structure, arms A and B as stored, no scan.
C  test correlations and private sampling SE of the pick-2 pairs."""
import sys
sys.path.insert(0, '<repo>/reports/TRICKS_1115/critic_checks')
from critic_lib import *
from math import erf, sqrt
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
L = []
def w_(s=''):
    print(s, flush=True); L.append(s)
big = Mv >= 50
# ---------------- A
w_('## A. domain token, re-derived from reports/TRICKS_1115/synth_checks/domtok/preds (CPU fp32 files of the synthesiser)')
D = R / 'reports/TRICKS_1115/synth_checks/domtok/preds'; mem = wide('fd10', S8); FDP = [f'abl_fd10_pop1kd05w192_s{s}' for s in S4]
zpop_to = z(pool(FDP)[1]); zft = z(np.load(P / 'ens_FD16P20kd_ft_val.npy'))
def tofile(kind):
    wp = np.mean([z(np.load(D / f'cpu_{m}_{kind}.npy')) for m in mem], 0); to = 0.8 * z(wp) + 0.2 * zpop_to
    return ndcg(0.5 * zft + 0.5 * z(to)), ndcg(to), ndcg(wp)
ref = tofile('base'); w_(f'   CPU reference: file {ref[0].mean():.5f} | train-only branch {ref[1].mean():.5f} | 16-pool of the wide members {ref[2].mean():.5f}')
for kind in ('prod', 'mix'):
    a = tofile(kind); d = a[0] - ref[0]
    w_(f'   {kind}: FILE {three(d)} | M<50 {f3(wm(d[~big]))} | M>=50 {f3(wm(d[big]))}')
    w_(f'        train-only branch {three(a[1] - ref[1])}')
    w_(f'        16 wide members alone {three(a[2] - ref[2])}')
    v, pv = wm(d), wm(d, WPRIV); m5 = wm(d[~big])
    w_(f'        rule clauses: val >= 0 {v[0] >= 0} | pmix-private >= +1 SE {pv[0] >= pv[1]} | M<50 >= -1 SE {m5[0] >= -m5[1]}')
# ---------------- B
w_(); w_('## B. lag-equalised inputs, re-derived from reports/TRICKS_1115/web_shift_probe/preds (4 held-out students kd05w192 s951-954, CPU fp32)')
E_ = R / 'reports/TRICKS_1115/web_shift_probe/preds'; m4 = [f'w5_e10_kd05w192_s{s}' for s in S4]; pop4 = [f'abl_e10_pop1kd05t25_s{s}' for s in S4]
pp = pool(pop4); zpf, zpt = z(pp[0]), z(pp[1]); ft4 = z(np.mean([mz(m)[0] for m in m4], 0)); ftb = 0.8 * ft4 + 0.2 * zpf
def eq(kind):
    wp = z(np.mean([z(np.load(E_ / f'cpu_{m}_{kind}.npy')) for m in m4], 0))
    return ndcg(wp), ndcg(0.5 * z(ftb) + 0.5 * z(0.8 * wp + 0.2 * zpt)), ndcg(0.5 * z(ft4) + 0.5 * wp)
b0 = eq('base'); w_(f'   CPU baseline: 4-pool train-only {b0[0].mean():.5f} | 4 + 4 file (0.5 z(FT stored) + 0.5 z(0.8 wide + 0.2 popularity-light)) {b0[1].mean():.5f} | 4-member file without popularity-light students {b0[2].mean():.5f}')
for kind, nm in (('A', 'full dose'), ('B', 'half dose')):
    a = eq(kind)
    w_(f'   arm {kind} ({nm}): 4-pool train-only {three(a[0] - b0[0])}')
    d = a[1] - b0[1]; v, pb, pv = wm(d), wm(d, WPUB), wm(d, WPRIV)
    w_(f'        4 + 4 FILE {three(d)} | z val {v[0] / v[1]:+.2f}, pmix-private {pv[0] / pv[1]:+.2f} | M<50 {f3(wm(d[~big]))} | M>=50 {f3(wm(d[big]))} | paired public SE {d.std() / np.sqrt(NPUB):.5f}')
    w_(f'        4-member FILE without popularity-light students {three(a[2] - b0[2])}')
    w_(f'        step-2 clauses on this preview: val >= 0 {v[0] >= 0} | pmix-private >= +1 SE {pv[0] >= pv[1]} | M<50 >= -1 SE {wm(d[~big])[0] >= -wm(d[~big])[1]}')
    if kind == 'B':
        se = pv[1]
        for t in (0.0, 0.0001, 0.0002, 0.0003): w_(f'        power of "pmix-private >= +1 SE" at this SE ({se:.5f}): true file effect {t:+.4f} -> P(clause holds) {1 - Phi(1 - t / se):.2f}')
# ---------------- C
w_(); w_('## C. pick-2 pairs: test correlation and sampling SE of the difference (sd of the val per-feed delta / sqrt(n))')
def filez(base, pop, sp): return fin(branches(base, pop, 0.2, sp), sp)
A8 = (wide('fd10', S4), FDP); B8 = (wide('fd10', S4b), [f'abl_fd10_pop1kd05w192_s{s}' for s in S4b])
FD16P20 = (wide('fd10', S8), FDP); W8P20 = (wide('e10', S4), pop4); VF = (wide('fd10', S8), FDP + [f'abl_fd10_pfkd05w192_s{s}' for s in S4])
def pair(nm, a, b):
    d = ndcg(filez(*a, 'val')) - ndcg(filez(*b, 'val')); c = np.corrcoef(filez(*a, 'test'), filez(*b, 'test'))[0, 1]
    sp = d.std() / np.sqrt(NPRIV); w_(f'   {nm}: val {f3(wm(d))} | test corr {c:.5f} | sampling SE public {d.std() / np.sqrt(NPUB):.5f}, private {sp:.5f} | 0.40 x private SE {0.3989 * sp:+.5f}')
pair('W8P20 - FD16P20', W8P20, FD16P20); pair('FD16P20vf - FD16P20', VF, FD16P20); pair('B8 (fd10 s955-958 + pop s955-958) - A8 (FD8P20)', B8, A8)
open(OUT / 'c3_synth_reads.txt', 'w').write('\n'.join(L) + '\n')
