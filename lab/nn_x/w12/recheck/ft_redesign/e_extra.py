"""E. (1) hard-label twins of the students (w5_e10_base_s951-954, same recipe/seeds, 1-seed FT n2cpv3): FT gain;
 (2) member-level linear shrink of the FT shift, z_TO + lam * s_m (W8 members), vs the logged epoch-1 gain;
 (3) pool proxy: z(TO pool) + one member's shift vs the true FT pool (is a 1-2 member pilot informative at pool level?);
 (4) is the fold-specific part of the test shift common to all members (same GroupKFold partition)?
 (5) quadratic model of NDCG(TO + lam * shift) at pool level, pooled over disjoint pools."""
from lib import *
print('== 1. hard-label twins (w5_e10_base, 1-seed FT) vs students at the same seeds, FT OOF - TO, paired per feed')
D = {}
for s in sd4:
    b = f'w5_e10_base_s{s}'; D[b] = ev(np.load(W / f'nn_{b}_n2cpv3_val_oof.npy')) - ev(np.load(W / f'nn_{b}_trainonly_val.npy'))
    for a in ('kd05', 'kd05t25', 'kd05w192', 'kd05t25w192', 'kd08'):
        m = f'w5_e10_{a}_s{s}'; D[m] = ev(raw(m, 'ft', 'val')) - ev(raw(m, 'to', 'val'))
db = np.mean([D[f'w5_e10_base_s{s}'] for s in sd4], 0); print(f'  base (hard labels, d128): {f(db)} | M<50 {f(db[Mv<50])} | M>=50 {f(db[Mv>=50])}')
for a in ('kd05', 'kd05t25', 'kd08', 'kd05w192', 'kd05t25w192'):
    d = np.mean([D[f'w5_e10_{a}_s{s}'] for s in sd4], 0); print(f'  {a:12s}: {f(d)} | minus base twin {f(d - db)} (M<50 {f((d-db)[Mv<50])}, M>=50 {f((d-db)[Mv>=50])})')
dm = np.mean([ev(raw(m, 'ft', 'val')) - ev(raw(m, 'to', 'val')) for m in GROUPS['H:m02']], 0)
print(f'  m02 x7 (re-cooled F7 roots, hard labels): {f(dm)} | base twin minus m02: {f(db - dm)}')
print('\n== 2. member level, W8: NDCG(z_TO + lam * s_m) - NDCG(z_TO), mean over 8 members (paired SE)')
lams = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
E0 = {m: ev(z(m, 'to', 'val')) for m in W8}
for lam in lams:
    d = np.mean([ev(z(m, 'to', 'val') + lam * (z(m, 'ft', 'val') - z(m, 'to', 'val'))) - E0[m] for m in W8], 0); print(f'  lam {lam:.2f}: {f(d)}')
print('  (logged raw-weight epoch-1 gain, W8 mean: +0.00047; saved EMA-final: -0.00043)')
print('\n== 3. pool proxy: z(TO pool W8) + s_m (ONE member shift) vs the true FT pool W8 and vs the TO pool')
pt, pf = zv(pool(W8, 'to', 'val')), zv(pool(W8, 'ft', 'val')); e_to, e_ft = ev(pt), ev(pf)
pr = []
for m in W8:
    e = ev(pt + z(m, 'ft', 'val') - z(m, 'to', 'val')); pr.append(e)
    print(f'  {m:28s} proxy - TOpool {f(e - e_to)} | proxy - true FTpool {f(e - e_ft)}')
print(f'  mean of 8 single-member proxies - TO pool: {f(np.mean(pr, 0) - e_to)} ; true FT pool - TO pool: {f(e_ft - e_to)}')
for lam in (0.25, 0.5):
    e2 = np.mean([ev(pt + lam * (z(m, 'ft', 'val') - z(m, 'to', 'val'))) for m in W8[:1] + W8[4:5]], 0); et = ev(pt + lam * (pf - pt))
    print(f'  lam {lam}: 2-member proxies (kd05w192_s951, kd05t25w192_s951) - TO pool {f(e2 - e_to)} ; true pooled shift {f(et - e_to)}')
print('\n== 4. fold-specific part of the test shift: common to members? (test_folds; residual of fold k = shift_k - mean_k shift)')
R = {}
for m in W8:
    tf = np.load(W / f'nn_{m}_n2cpv3s3_test_folds.npy').astype(np.float64); Z = np.array([zt(x) for x in tf]); R[m] = Z - Z.mean(0)
    if m == W8[0]: print(f'  fold-residual RMS (member {m}): {np.sqrt((R[m]**2).mean()):.4f}; full single-fold shift RMS {np.sqrt(((Z - z(m,"to","test"))**2).mean()):.4f}')
cs = [np.corrcoef(R[a][k], R[b][k])[0, 1] for i, a in enumerate(W8) for b in W8[i + 1:] for k in range(5)]
cx = [np.corrcoef(R[a][k], R[b][(k + 1) % 5])[0, 1] for i, a in enumerate(W8) for b in W8[i + 1:] for k in range(5)]
print(f'  corr of fold-k residuals between two members: same fold {np.mean(cs):.3f} | different folds {np.mean(cx):.3f}')
Rp = np.mean([R[m] for m in W8], 0); print(f'  pooled (8 members) fold-residual RMS {np.sqrt((Rp**2).mean()):.4f} -> on test the 5-fold mean removes it; on val OOF it is in every prediction')
print('\n== 5. quadratic model d(lam) = a*lam - b*lam^2 of NDCG(z(TO pool) + lam * pooled shift) - NDCG(TO pool); lam grid 0..1 step 0.1 (least squares)')
ws = np.arange(0.1, 1.01, 0.1); A = np.stack([ws, -ws ** 2], 1)
tot = []
for k, ms in (('W8', W8), ('N8', N8), ('C3', C3M), ('kd08', GROUPS['S:kd08'])):
    a_, b_ = zv(pool(ms, 'to', 'val')), zv(pool(ms, 'ft', 'val')); e0 = ev(a_)
    dl = np.array([ev(a_ + w * (b_ - a_)) - e0 for w in ws]); tot.append(dl)
    co = np.linalg.lstsq(A, dl.mean(1), rcond=None)[0]
    print(f'  {k:5s} a {co[0]:+.5f} b {co[1]:+.5f} -> lam* {co[0]/(2*co[1]) if co[1] > 0 else float("nan"):.2f}, max gain {co[0]**2/(4*co[1]) if co[1] > 0 else float("nan"):+.5f}; d(0.25) {f(dl[1]*0.5+dl[2]*0.5)} d(0.5) {f(dl[4])} d(1) {f(dl[9])}')
T = np.mean(tot, 0); co = np.linalg.lstsq(A, T.mean(1), rcond=None)[0]
# bootstrap over feeds for lam* and the max gain
rng = np.random.default_rng(0); bs = []
for _ in range(2000):
    i = rng.integers(0, T.shape[1], T.shape[1]); c = np.linalg.lstsq(A, T[:, i].mean(1), rcond=None)[0]
    bs.append((c[0], c[1], c[0] / (2 * c[1]) if c[1] > 0 else np.nan, c[0] ** 2 / (4 * c[1]) if c[1] > 0 and c[0] > 0 else 0.0))
bs = np.array(bs)
print(f'  mean of the 4 pools: a {co[0]:+.5f} (boot sd {bs[:,0].std():.5f}) b {co[1]:+.5f} (sd {bs[:,1].std():.5f}); max gain over lam {co[0]**2/(4*co[1]):+.5f} (boot mean {bs[:,3].mean():+.5f}, 95% upper {np.quantile(bs[:,3], 0.95):+.5f})')
print(f'  if the noise term b were halved / quartered with a unchanged: best gain {co[0]**2/(2*co[1]):+.5f} / {co[0]**2/(co[1]):+.5f}; upper-95% a: {np.quantile(bs[:,0],0.95):+.5f} -> gain at b/4: {np.quantile(bs[:,0],0.95)**2/co[1]:+.5f}')
