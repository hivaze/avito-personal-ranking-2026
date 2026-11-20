"""Read-only CPU check 2 (rule written before the run, see RULE below).
Softmax-probability pooling (T = 1, no scan) against the file form (per-feed z-mean) on the HELD-OUT twin members:
16 wide students w5_e10_{kd05w192,kd05t25w192}_s951-958 + 4 wide popularity-light students abl_e10_pop1kd05w192_s951-954.
Witness 1 (fresh): train-only pools on the 30,000 holdout feeds, selected checkpoints, 0.8 z(wide) + 0.2 z(pop).
Witness 2 (repeat look; same val feeds as check 1 on the full-data twins): final val of the W16P20w form.
RULE (declared before any number of this script): softmax pooling is SUPPORTED iff holdout delta >= +2 SE AND val delta >= 0; else NOT SUPPORTED.
Looks so far: 1 (check 1, full-data members, val: +0.00019 +- 0.00013; three alternative forms were tried there)."""
import sys, datetime
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
print('run at', datetime.datetime.utcnow().isoformat() + 'Z')
m = Meta()
WIDE = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 959)]; POP = [f'abl_e10_pop1kd05w192_s{s}' for s in S4]
def sm(x, off, M):
    x = np.asarray(x, np.float64); fid = np.repeat(np.arange(len(M)), M); mx = np.maximum.reduceat(x, off); e = np.exp(x - mx[fid]); s = np.add.reduceat(e, off); return e / s[fid]
se = lambda d: d.std() / np.sqrt(len(d))
def forms(sp):
    off, M = m.off[sp], m.M[sp]
    return {'zmean': lambda x: m.z(sp, x), 'softmaxmean': lambda x: sm(x, off, M), 'rawmean': lambda x: np.asarray(x, np.float64), 'rankmean': lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 0]}
# witness 1: holdout, train-only
H = {}
for v in WIDE + POP:
    H[v] = np.load(paths(v)[1]).astype(np.float32)
res = {}
for name, f in forms('hold').items():
    ew = np.mean([f(H[v]) for v in WIDE], 0); ep = np.mean([f(H[v]) for v in POP], 0)
    res[name] = (ndcg_fast(0.8 * m.z('hold', ew) + 0.2 * m.z('hold', ep), m.yh, m.oh, m.Mh), ndcg_fast(m.z('hold', ew), m.yh, m.oh, m.Mh))
print('HOLDOUT (30,000 feeds; train-only pools)')
for name in res:
    d = res[name][0] - res['zmean'][0]; dw = res[name][1] - res['zmean'][1]
    print(f'  {name:12s} 16+4 pool {res[name][0].mean():.5f} (vs zmean {d.mean():+.5f} +- {se(d):.5f}) | wide-16 pool alone {res[name][1].mean():.5f} ({dw.mean():+.5f} +- {se(dw):.5f})')
dh = res['softmaxmean'][0] - res['zmean'][0]
# witness 2: val, final form (FT out-of-fold + train-only)
PR = ROOT / 'preds'
def lv(v, suf): return np.load(PR / f'nn_{v}_{suf}.npy').astype(np.float32)
rv = {}
for name, f in forms('val').items():
    br = []
    for suf in ('n2cpv3s3_val_oof', 'trainonly_val'):
        ew = np.mean([f(lv(v, suf)) for v in WIDE], 0); ep = np.mean([f(lv(v, suf)) for v in POP], 0)
        br.append(0.8 * m.z('val', ew) + 0.2 * m.z('val', ep))
    rv[name] = ndcg_fast(0.5 * m.z('val', br[0]) + 0.5 * m.z('val', br[1]), m.yv, m.ov, m.Mv)
print('VAL (14,423 feeds; final form, held-out members)')
for name in rv:
    d = rv[name] - rv['zmean']; big = m.Mv >= 50
    print(f'  {name:12s} final {rv[name].mean():.5f} (vs zmean {d.mean():+.5f} +- {se(d):.5f}; M>=50 {d[big].mean():+.5f} +- {se(d[big]):.5f}, M<50 {d[~big].mean():+.5f} +- {se(d[~big]):.5f})')
dv = rv['softmaxmean'] - rv['zmean']
ok = dh.mean() >= 2 * se(dh) and dv.mean() >= 0
print(f'RULE: holdout {dh.mean():+.5f} +- {se(dh):.5f} (z {dh.mean() / se(dh):+.2f}), val {dv.mean():+.5f} +- {se(dv):.5f} -> softmax pooling', 'SUPPORTED' if ok else 'NOT SUPPORTED')
