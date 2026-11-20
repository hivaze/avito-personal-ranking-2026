"""M2: does a popularity-light partner pool that imitated a full-view teacher less make a better 0.2 admixture partner?
Base = W16 train-only pool (16 held-out wide students, selected checkpoints). Partner pools (all held-out, train-only, selected checkpoints):
  HL4 / HL8 = hard-label pop1 members abl_e10_pop1_s951-954 / s951-958 (d128, no teacher)
  KDn4      = abl_e10_pop1kd05t25_s951-954 (d128, teacher_mean25: 8 of its 25 members are pop1)
  KDw4      = abl_e10_pop1kd05w192_s951-954 (d192, teacher_mean11: full-view teacher only)
Reads: member / pool NDCG, correlation of the partner pool with the base pool, admixture delta (1-w) z(base) + w z(partner) minus base
on val (plain, pmix-public, pmix-private weights from src/pmix.py) and on the in-period holdout (reported; biased against the view).
CPU only, read-only on every prediction file."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
import pmix
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
Wt = pmix.weights(); wpub, wpriv = np.asarray(Wt['public'], float), np.asarray(Wt['private'], float)
def wm(d, w):
    mu = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum(); return mu, np.sqrt(np.average((d - mu) ** 2, weights=w) / neff)
def zz(v):
    pv, ph, _ = paths(v, ''); return m.z('val', np.load(pv).astype(np.float32)), m.z('hold', np.load(ph).astype(np.float32))
base = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 959)]
part = dict(HL4=[f'abl_e10_pop1_s{s}' for s in range(951, 955)], HL8=[f'abl_e10_pop1_s{s}' for s in range(951, 959)],
            HL4b=[f'abl_e10_pop1_s{s}' for s in range(955, 959)],
            KDn4=[f'abl_e10_pop1kd05t25_s{s}' for s in range(951, 955)], KDw4=[f'abl_e10_pop1kd05w192_s{s}' for s in range(951, 955)])
B = [zz(v) for v in base]; bv = m.z('val', np.mean([b[0] for b in B], 0)); bh = m.z('hold', np.mean([b[1] for b in B], 0))
nbv, nbh = ev(bv), eh(bh)
print(f'base W16 TO pool: val {nbv.mean():.5f} holdout {nbh.mean():.5f}')
cz = lambda a, b: float(np.mean(a * b) / np.sqrt(np.mean(a * a) * np.mean(b * b)))
D = {}
for k, mem in part.items():
    Zs = [zz(v) for v in mem]; pv = m.z('val', np.mean([z[0] for z in Zs], 0)); ph = m.z('hold', np.mean([z[1] for z in Zs], 0))
    memv = np.mean([ev(z[0]).mean() for z in Zs]); memh = np.mean([eh(z[1]).mean() for z in Zs])
    within = np.mean([cz(Zs[i][0], Zs[j][0]) for i in range(len(Zs)) for j in range(i + 1, len(Zs))])
    print(f'\n{k}: member val {memv:.5f} holdout {memh:.5f} | pool val {ev(pv).mean():.5f} holdout {eh(ph).mean():.5f} | within-pool corr {within:.4f} | corr(pool, base pool) val {cz(pv, bv):.4f} holdout {cz(ph, bh):.4f}'
          f' | mean member corr to base pool {np.mean([cz(z[0], bv) for z in Zs]):.4f}')
    for w in (0.1, 0.2, 0.3):
        dv = ev((1 - w) * bv + w * pv) - nbv; dh = eh((1 - w) * bh + w * ph) - nbh; D[k, w] = (dv, dh)
        a, b = wm(dv, wpub); c, d = wm(dv, wpriv)
        print(f'   w {w}: val {fmt(dv)} | pmix-public {a:+.5f}+-{b:.5f} | pmix-private {c:+.5f}+-{d:.5f} | holdout {fmt(dh)}')
print('\npaired differences between partner pools at w 0.2 (val | pmix-private | holdout):')
for a, b in (('HL4', 'KDn4'), ('HL8', 'KDn4'), ('HL4b', 'KDn4'), ('KDw4', 'KDn4'), ('HL8', 'KDw4')):
    dv = D[a, 0.2][0] - D[b, 0.2][0]; dh = D[a, 0.2][1] - D[b, 0.2][1]; c, d = wm(dv, wpriv)
    print(f'   {a} - {b}: val {fmt(dv)} | pmix-private {c:+.5f}+-{d:.5f} | holdout {fmt(dh)}')
