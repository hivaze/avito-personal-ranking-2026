"""Staleness-family read-out ("RULE hz", nn_x/w5/logs/E_kd_predeclared.txt 2026-11-15): train-only pools, arm vs reference, selected AND @last.
  re-cool arms : hz_read.py rc:<arm> [ref arm = ref]            (members = K and L snapshots of seeds 951-958 whose arm AND ref runs are complete; paired by snapshot)
  from scratch : hz_read.py <arm> <seeds csv> [ref arm = kd05w192]
Columns: val (all, M>=50, M<50), pmix-public / pmix-private (val reweighted, same sample), stale-val (nn_x/w5/preds/stale/nn_<v>[@last]_staleval.npy, if every member has one),
in-period holdout (REPORTED ONLY for this family). RULE: PASS iff val >= 0 (hz2-type arms: >= -1 SE) AND stale-val >= +2 SE AND val M<50 >= -1 SE; DECISION on the @last checkpoints (amendment 2026-11-15T04:57Z), selected reported. CPU, read-only."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition'); sys.path.insert(0, '<repo>/src')
from pc_common import *
import pmix
a = sys.argv[1]; m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh); wts = pmix.weights(); big = m.Mv >= 50
done = lambda v: (ROOT / f'nn_x/w5/logs/{v}.json').exists() and (W / f'holdout/nn_{v}@last_holdout.npy').exists()
if a.startswith('rc:'):
    arm = a[3:]; ref = sys.argv[2] if len(sys.argv) > 2 else 'ref'
    P_ = [(f'w5_{st}cd_{arm}_s{x}', f'w5_{st}cd_{ref}_s{x}') for st in 'kl' for x in range(951, 959)]
else:
    arm = a; seeds = [int(x) for x in sys.argv[2].split(',')]; ref = sys.argv[3] if len(sys.argv) > 3 else 'kd05w192'
    P_ = [(f'w5_e10_{arm}_s{x}', f'w5_e10_{ref}_s{x}') for x in seeds]
P_ = [(x, y) for x, y in P_ if done(x) and done(y)]; assert P_, 'no complete (arm, ref) pair'
print(f'== {arm} vs {ref}: {len(P_)} pairs: {" ".join(x[-8:] for x, _ in P_)}')
res = {}
for kind in ('', '@last'):
    def pool(names, sp, fn):
        return np.mean([m.z(sp, np.load(fn(v)).astype(np.float32)) for v in names], 0)
    A, R = [x for x, _ in P_], [y for _, y in P_]
    fv = lambda v: W / f'nn_{v}{kind}_trainonly_val.npy'; fh = lambda v: W / f'holdout/nn_{v}{kind}_holdout.npy'; fs = lambda v: W / f'stale/nn_{v}{kind}_staleval.npy'
    dv = ev(pool(A, 'val', fv)) - ev(pool(R, 'val', fv)); dh = eh(pool(A, 'hold', fh)) - eh(pool(R, 'hold', fh))
    mem = np.mean([ev(m.z('val', np.load(fv(x)))).mean() - ev(m.z('val', np.load(fv(y)))).mean() for x, y in P_])
    pp, sp_ = pmix.wmean_se(dv, wts['public']); pr, sr = pmix.wmean_se(dv, wts['private'])
    line = (f'  {kind or "selected":8s}: val {fmt(dv)} (z {dv.mean() / se(dv):+.2f}; M>=50 {fmt(dv[big])}, M<50 {fmt(dv[~big])}; member mean {mem:+.5f}) | pmix-public {pp:+.5f}+-{sp_:.5f} pmix-private {pr:+.5f}+-{sr:.5f}'
            f' | holdout (reported) {fmt(dh)}')
    st = None
    if all(fs(v).exists() for v in A + R):
        ds = ev(pool(A, 'val', fs)) - ev(pool(R, 'val', fs)); st = (ds.mean(), se(ds)); dd = ds - dv
        line += f' | STALE-VAL {fmt(ds)} (z {ds.mean() / se(ds):+.2f}; M>=50 {fmt(ds[big])}, M<50 {fmt(ds[~big])}); stale minus fresh {fmt(dd)}; levels arm {ev(pool(A, "val", fs)).mean():.5f} ref {ev(pool(R, "val", fs)).mean():.5f}'
    else: line += ' | STALE-VAL: files missing'
    print(line); res[kind] = (dv, st)
    if st is not None:
        h2 = arm.startswith(('hz2', 'h2'))  # hz2-type arms: val >= -1 SE (declared 2026-11-15T04:57Z); others: val >= 0
        vok = dv.mean() >= (-se(dv) if h2 else 0.0)
        ok = vok and st[0] >= 2 * st[1] and dv[~big].mean() >= -se(dv[~big])
        print(f'           RULE on {kind or "selected"}: val {">= -1 SE" if h2 else ">= 0"} {vok}; stale-val >= +2 SE {st[0] >= 2 * st[1]} (z {st[0] / st[1]:+.2f}); M<50 >= -1 SE {dv[~big].mean() >= -se(dv[~big])} -> {"PASS" if ok else "no pass"}'
              + ('  <== DECISION COLUMN (@last; amendment 04:57Z)' if kind == '@last' else '  (reported)'))
