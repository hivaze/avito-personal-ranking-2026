"""premortem t9_softft: re-read of the stored soft-target fine-tune pilot (ft_redesign section 8) at the private mix. PREDECLARED block 6. Writes out/t9_softft.md."""
from pm_common import *
m = Meta(test=False, hold=False); Wt = m.pmix(); wpriv = Wt['private']
FR = ROOT / 'nn_x/w12/recheck/ft_redesign/out'
Mv, ov, yv = m.M['val'], m.off['val'], m.y['val']
ptv = np.load(CACHE / 'pool_val_to_W8.npy')          # member-mean of z of the 8 W8 train-only members (= the pilot's TO pool)
members = ['w5_e10_kd05w192_s951', 'w5_e10_kd05t25w192_s951']
arms = {mm: {t: (np.load(FR / f'ftcpu_{mm}_{t}_oof.npy'), np.load(FR / f'ftcpu_{mm}_{t}_done.npy').astype(bool)) for t in ('A_base', 'B_kd05')} for mm in members}
dn = np.all([arms[mm][t][1] for mm in members for t in ('A_base', 'B_kd05')], 0)
rows = np.repeat(dn, Mv); Ms = Mv[dn]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64); ys = yv[rows]
from pm_common import _ndcg, _feed_z
Z = lambda x: _feed_z(offs, Ms, np.ascontiguousarray(np.asarray(x, np.float32)[rows] if len(x) == len(rows) else np.asarray(x, np.float32))).astype(np.float64)
N = lambda s: _ndcg(np.ascontiguousarray(s, np.float32), ys, offs, Ms)
zpt = Z(ptv); e_pt = N(zpt)
fileproxy = {}; member = {}
for mm in members:
    for t in ('A_base', 'B_kd05'):
        O = arms[mm][t][0]; E = O.shape[0] - 2; za = Z(O[E + 1]); zto = Z(O[0]); prox = zpt + (za - zto)
        zp = _feed_z(offs, Ms, prox.astype(np.float32)).astype(np.float64)
        fileproxy[(mm, t)] = N(0.5 * zp + 0.5 * zpt) - e_pt; member[(mm, t)] = N(za) - N(zto)
d = np.mean([fileproxy[(mm, 'B_kd05')] - fileproxy[(mm, 'A_base')] for mm in members], 0)
dm = np.mean([member[(mm, 'B_kd05')] - member[(mm, 'A_base')] for mm in members], 0)
a = np.mean([fileproxy[(mm, 'A_base')] for mm in members], 0)
w = wpriv[dn]; c = m.vcold[dn]; big = Mv[dn] >= 50
out = []
def P_(s=''):
    print(s, flush=True); out.append(s)
P_(f'## T9. Soft-target fine-tune pilot re-read ({int(dn.sum())} feeds done by all arms of both members; final EMA; mean of the two members; paired per feed)')
P_('| quantity | plain val (pilot: see right) | pmix-private weights | cold share >= 0.30 | cold share < 0.30 | M >= 50 | M < 50 |')
P_('|---|---|---|---|---|---|---|')
for lab, x, ref in (('file proxy, B soft targets - A baseline', d, '-0.00036 ± 0.00031'), ('member - TO, B - A', dm, '+0.00053 ± 0.00038'), ('file proxy of the baseline A - TO pool', a, '+0.00054 ± 0.00053')):
    P_(f'| {lab} | {f5(wms(x))} (pilot {ref}) | {f5(wms(x, w))} | {f5(wms(x[c >= 0.30]))} | {f5(wms(x[c < 0.30]))} | {f5(wms(x[big]))} | {f5(wms(x[~big]))} |')
v = wms(d); ok = abs(v[0] - (-0.00036)) <= 0.00003
r1 = wms(d, w); r2 = wms(d[c >= 0.30])
P_(f'CONTROL (plain val within 0.00003 of -0.00036): {"VALID" if ok else "FAIL"}. RULE (block 6): pmix-private {r1[0] / r1[1]:+.2f} SE, cold >= 0.30 {r2[0] / r2[1]:+.2f} SE -> '
   + ('RE-OPEN CANDIDATE' if (ok and (r1[0] >= r1[1] or r2[0] >= r2[1])) else 'the closure also holds at the private mix (same low power)' if ok else 'no read (control failed)') + '.')
open(OUT / 't9_softft.md', 'w').write('\n'.join(out) + '\n')
