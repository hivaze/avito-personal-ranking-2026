"""[W8 + extra members] vs W8, read directly from prediction files (no cache). CPU, read-only.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/recheck/pool_composition/pc_pool_vs_w8.py <variant,variant,...> [only]
  variants = full names (w5_e10_kd05w192_s956, abl_e10_...); 'only' = pool of the listed members alone (without W8).
Prints: TO pool val / holdout / holdout @last deltas vs W8 (paired per feed), val slices, the final 0.5 z(FT W8) + 0.5 z(TO pool) on val, test correlations.
Reference: W8 TO 0.40157 | 0.32456, final 0.40165; saturation law b (1/8 - 1/(8+k)) with b 0.0025 val | 0.0021 hold for k extra seeds of the same family."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
extra = sys.argv[1].split(','); only = len(sys.argv) > 2 and sys.argv[2] == 'only'
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh); big = m.Mv >= 50
zv = lambda x: m.z('val', x); zt = lambda x: m.z('test', x)
def ld(v, suf=''):
    pv, ph, pt = paths(v, suf); return m.z('val', np.load(pv)), m.z('hold', np.load(ph)), m.z('test', np.load(pt))
W8n = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in (951, 952, 953, 954)]
W = [ld(v) for v in W8n]; WL = [ld(v, '@last') for v in W8n]; X = [ld(v) for v in extra]
haslast = all(paths(v, '@last')[1].exists() for v in extra); XL = [ld(v, '@last') for v in extra] if haslast else None
pool = lambda A, i: np.mean([a[i] for a in A], 0); S = X if only else W + X
ftv = np.mean([zv(np.load(P/f'nn_{v}_n2cpv3s3_val_oof.npy')) for v in W8n], 0); ftt = np.mean([zt(np.load(P/f'nn_{v}_n2cpv3s3_test.npy')) for v in W8n], 0)
dv = ev(pool(S, 0)) - ev(pool(W, 0)); dh = eh(pool(S, 1)) - eh(pool(W, 1))
print(f'{"pool of" if only else "W8 +"} {len(extra)} members: TO val {ev(pool(S, 0)).mean():.5f} ({fmt(dv)}; M>=50 {fmt(dv[big])}, M<50 {fmt(dv[~big])}) | TO holdout {eh(pool(S, 1)).mean():.5f} ({fmt(dh)})' +
      (f' | holdout @last {fmt(eh(pool(XL if only else WL + XL, 1)) - eh(pool(WL, 1)))}' if haslast else ''))
f0 = 0.5 * zv(ftv) + 0.5 * zv(pool(W, 0)); f1 = 0.5 * zv(ftv) + 0.5 * zv(pool(S, 0)); d = ev(f1) - ev(f0)
t0 = 0.5 * zt(ftt) + 0.5 * zt(pool(W, 2)); t1 = 0.5 * zt(ftt) + 0.5 * zt(pool(S, 2))
print(f'final 0.5 z(FT W8) + 0.5 z(TO pool): val {ev(f1).mean():.5f} ({fmt(d)}) | test corr TO pool {np.corrcoef(pool(S, 2), pool(W, 2))[0,1]:.4f}, final {np.corrcoef(t1, t0)[0,1]:.4f}')
for v, x in zip(extra, X): print(f'   {v:32s} val {ev(x[0]).mean():.5f} hold {eh(x[1]).mean():.5f} corr to W8 pool (val) {np.corrcoef(x[0], pool(W, 0))[0,1]:.4f}')
