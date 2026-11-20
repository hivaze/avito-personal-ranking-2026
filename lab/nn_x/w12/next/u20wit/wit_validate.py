"""u20wit step 3: validation of the CPU inference path and of the witness on checkpoints with KNOWN answers (run BEFORE any U20 read).
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/wit_validate.py
Needs cache/sc/w5_e10_{xf0,base,xf1}_s95{1,2,3}__tr.npy (wit_score.py tr W0S,H0,W0L,W1S,W1L ...).
(a) CPU logits of the 3 hard-label fold-0-excluded members vs nn_x/w12/kd/out/teacher_xf0.npy (their GPU bf16 SUM on every train row): within-feed correlation;
    plus per member on H0: CPU vs the stored GPU holdout file.
(b) base - xf0 (in-sample margin on fold-0 feeds, ~0 on fold-1 feeds), and the fold swap xf1 - xf0 (sign must flip between fold-0 and fold-1 feeds).
(c) positives per feed by M bin: natural short witness vs val M < 50.
(d) resolution of the witness: seed-to-seed (A/A) per-feed sd of a member delta, and the SE this implies for a 2-seed pooled B - A.
Labels: train labels of the witness rows and val labels are used for evaluation only."""
from wit_common import *
sys.path.insert(0, str(ROOT / 'nn_x/w12/recheck/pool_composition'))
import numba as nb
from stack_val import feed_rank_z
sm = np.load(CACHE / 'sub_meta.npz'); sub = CACHE / 'sub'
M = sm['M'].astype(np.int64); off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64); rows = np.load(sub / 'rows.npy'); y = np.load(sub / 'cand_target.npy').astype(np.float32)
fid = np.repeat(np.arange(len(M)), M); sid = sm['set_id']; npos = np.bincount(fid, weights=y, minlength=len(M))
S3 = (951, 952, 953)
sc = lambda v, dom='tr': np.load(CACHE / 'sc' / f'{v}__{dom}.npy')
z = lambda x: feed_rank_z(off, M, np.nan_to_num(np.asarray(x, np.float32)))[:, 1]
E = lambda x: ndcg_feeds(np.nan_to_num(x), y, off, M)
def wcorr(a, b):
    """within-feed Pearson correlation per feed (float64)."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    ca = a - (np.bincount(fid, weights=a) / M)[fid]; cb = b - (np.bincount(fid, weights=b) / M)[fid]
    return np.bincount(fid, weights=ca * cb) / np.sqrt(np.bincount(fid, weights=ca * ca) * np.bincount(fid, weights=cb * cb) + 1e-30)
print(f'witness sub-split: {len(M)} feeds, {len(y)} rows; feeds without a positive: {int((npos == 0).sum())}')
print('\n(a) CPU fp32 path vs the stored teacher (GPU bf16 SUM of 3 members, dump_teacher.py)')
n_t = int(open(ROOT / 'nn_x/w12/kd/out/teacher_xf0.n').read()); T = np.asarray(np.load(ROOT / 'nn_x/w12/kd/out/teacher_xf0.npy', mmap_mode='r')[rows], np.float32) / n_t
X0 = [sc(f'w5_e10_xf0_s{s}') for s in S3]; mine = np.mean(X0, 0)
print(f'  teacher_xf0.npy: SUM of {n_t} members (file .n); compared as mean logit')
c = wcorr(mine, T)
print(f'  {"set":4s} {"feeds":>6s} {"rows":>8s} | within-feed corr(mean of my 3 CPU members, teacher/3): mean / median / p1 / min | share >= 0.999 | centred row corr | NDCG mine / teacher (paired)')
for j, k in enumerate(SETS):
    f = sid == j; r = f[fid]; a, b = mine[r].astype(np.float64), T[r].astype(np.float64)
    ca = a - (np.bincount(fid[r], weights=a, minlength=len(M)) / M)[fid[r]]; cb = b - (np.bincount(fid[r], weights=b, minlength=len(M)) / M)[fid[r]]
    em, et = E(mine)[f], E(T)[f]
    print(f'  {k:4s} {int(f.sum()):6d} {int(r.sum()):8d} | {c[f].mean():.6f} / {np.median(c[f]):.6f} / {np.quantile(c[f], 0.01):.6f} / {c[f].min():.6f} | {np.mean(c[f] >= 0.999):.4f} | {np.corrcoef(ca, cb)[0, 1]:.6f} | {em.mean():.5f} / {et.mean():.5f} ({fmt(em - et)})')
print(f'  single member vs teacher mean (must be lower: the teacher is a 3-mean): ' + ', '.join(f's{s} {wcorr(x, T).mean():.4f}' for s, x in zip(S3, X0)))
print(f'  wrong-alignment control (my scores shifted by one feed against the teacher inside W0S): corr {wcorr(np.roll(mine, 31), T)[sid == 0].mean():.4f}')
# H0: per-member CPU vs stored GPU holdout predictions
tm = train_meta(); hold = tm['hold']; Mh = tm['M'][hold]; oh = np.concatenate([[0], np.cumsum(Mh)[:-1]]); h0 = np.where(tm['xf'][hold] == 0)[0]
assert (hold[h0] == sm['src_fidx'][sid == 2]).all()
hrows = np.concatenate([np.arange(oh[i], oh[i] + Mh[i]) for i in h0]); rH = (sid == 2)[fid]; fH = sid == 2
hm = np.load(W5P / 'holdout/holdout_meta.npy'); assert (hm[0] == Mh).all()
yh = np.load(W5P / 'holdout/holdout_y.npy'); assert (yh[hrows] == y[rH]).all(), 'holdout labels differ from the witness cache labels on H0'
print('  H0 per member, CPU fp32 vs stored GPU bf16 holdout file (5,974 fold-0 holdout feeds):')
for arm in ('xf0', 'base', 'xf1'):
    for s in S3:
        v = f'w5_e10_{arm}_s{s}'; p = CACHE / 'sc' / f'{v}__tr.npy'
        if not p.exists() or np.isnan(np.load(p)[rH]).any(): continue
        g = np.full(len(y), np.nan, np.float32); g[rH] = np.load(W5P / f'holdout/nn_{v}_holdout.npy')[hrows]; cpu = sc(v)
        ec, eg = E(cpu)[fH], E(g)[fH]; cc = wcorr(np.nan_to_num(cpu) * rH, np.nan_to_num(g) * rH)[fH]
        print(f'    {v}: CPU {ec.mean():.5f} GPU file {eg.mean():.5f} paired {fmt(ec - eg)} | within-feed corr mean {cc.mean():.6f} min {cc.min():.5f}')

print('\n(b) known-answer contrasts (CPU vs CPU, member deltas pooled over 3 seeds, paired per feed; 3-pool = mean per-feed z)')
def arm(a): return [sc(f'w5_e10_{a}_s{s}') for s in S3]
A0, B0 = X0, arm('base'); have1 = all((CACHE / 'sc' / f'w5_e10_xf1_s{s}__tr.npy').exists() and not np.isnan(sc(f'w5_e10_xf1_s{s}')).any() for s in S3); A1 = arm('xf1') if have1 else None
def contrast(X, Y, f):
    dm = np.mean([E(x)[f] - E(yy)[f] for x, yy in zip(X, Y)], 0)
    dp = E(np.mean([z(x) for x in X], 0))[f] - E(np.mean([z(yy) for yy in Y], 0))[f]
    return dm, dp
print(f'  {"set":4s} {"feeds":>6s} | xf0 member / 3-pool | base member / 3-pool | base - xf0: member, 3-pool' + (' | xf1 - xf0: member, 3-pool' if have1 else ''))
for j, k in enumerate(SETS):
    f = sid == j
    l0 = np.mean([E(x)[f].mean() for x in A0]); p0 = E(np.mean([z(x) for x in A0], 0))[f].mean(); l1 = np.mean([E(x)[f].mean() for x in B0]); p1 = E(np.mean([z(x) for x in B0], 0))[f].mean()
    dm, dp = contrast(B0, A0, f); t = f'  {k:4s} {int(f.sum()):6d} | {l0:.5f} / {p0:.5f} | {l1:.5f} / {p1:.5f} | {fmt(dm)}, {fmt(dp)}'
    if have1:
        dm, dp = contrast(A1, A0, f); t += f' | {fmt(dm)}, {fmt(dp)}'
    print(t)
print('  expected: base - xf0 > 0 on W0S / W0L (fold 0 is in-sample for base only), ~0 on W1S / W1L (in-sample for both), small on H0 (out-of-sample for both; same-user channel);')
print('            xf1 - xf0 > 0 on W0*, < 0 on W1* (fold swap).')
print('  by old-posterior class inside W0L (old p > 0.5 = in the cooldown pool of every full-data model):')
f = (sid == 1) & (sm['pold'] > 0.5); g = (sid == 1) & (sm['pold'] <= 0.5)
print(f'    base - xf0 member: pool feeds ({int(f.sum())}) {fmt(contrast(B0, A0, f)[0])} | non-pool feeds ({int(g.sum())}) {fmt(contrast(B0, A0, g)[0])}')

print('\n(c) positives per feed by M bin: natural short witness (W0S, W1S; train labels) vs val M < 50 (val labels; counts only)')
import polars as pl
fv = pl.read_parquet(FLAT / 'val/feeds.parquet'); Mv = fv['M'].to_numpy().astype(np.int64); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32)
pv = np.bincount(np.repeat(np.arange(len(Mv)), Mv), weights=yv, minlength=len(Mv))
p2 = np.load(PR_V2); pv2_all = np.nan_to_num(p2['p_random'].astype(np.float32), nan=0.0); isH = np.zeros(tm['n'], bool); isH[hold] = True
print(f'  {"M bin":7s} | {"W0S n":>6s} {"pos/feed":>8s} {"multi-pos":>9s} | {"W1S n":>6s} {"pos/feed":>8s} | {"val n":>6s} {"pos/feed":>8s} {"multi-pos":>9s} | skeptic NAT (all folds) / VAL')
ref = {(5, 10): (1.063, 1.078), (10, 20): (1.140, 1.157), (20, 35): (1.232, 1.250), (35, 50): (1.326, 1.359)}
for lo, hi in ((5, 10), (10, 20), (20, 35), (35, 50), (5, 50)):
    a = (sid == 0) & (M >= lo) & (M < hi); b = (sid == 3) & (M >= lo) & (M < hi); cv = (Mv >= lo) & (Mv < hi); r_ = ref.get((lo, hi), (float('nan'),) * 2)
    print(f'  {lo:2d}-{hi - 1:<4d} | {int(a.sum()):6d} {npos[a].mean():8.3f} {np.mean(npos[a] > 1):9.3f} | {int(b.sum()):6d} {npos[b].mean():8.3f} | {int(cv.sum()):6d} {pv[cv].mean():8.3f} {np.mean(pv[cv] > 1):9.3f} | {r_[0]:.3f} / {r_[1]:.3f}')
print(f'  M mix: W0S mean M {M[sid == 0].mean():.1f}, val M<50 mean M {Mv[Mv < 50].mean():.1f}; val M<5 feeds {int((Mv < 5).sum())}; W0L mean M {M[sid == 1].mean():.1f}, H0 {M[sid == 2].mean():.1f}, val M>=50 mean M {Mv[Mv >= 50].mean():.1f}')
print(f'  positives per feed, long: W0L {npos[sid == 1].mean():.3f}  H0 {npos[sid == 2].mean():.3f}  val M>=50 {pv[Mv >= 50].mean():.3f}')

print('\n(d) resolution: seed-to-seed (A/A) noise of a member delta on each set (xf0 seeds, out-of-sample on fold-0 sets)')
print(f'  {"set":4s} {"feeds":>6s} | per-feed sd of (member s_i - member s_j), mean of 3 pairs | implied SE of a 2-seed pooled member delta B - A (sd / sqrt(2) / sqrt(n)) | mean |A/A delta| of the 3 pairs')
EA = [E(x) for x in A0]
for j, k in enumerate(SETS):
    f = sid == j; sds = [np.std(EA[i][f] - EA[jj][f]) for i, jj in ((0, 1), (0, 2), (1, 2))]; mm = [abs(np.mean(EA[i][f] - EA[jj][f])) for i, jj in ((0, 1), (0, 2), (1, 2))]
    print(f'  {k:4s} {int(f.sum()):6d} | {np.mean(sds):.4f} | {np.mean(sds) / np.sqrt(2) / np.sqrt(f.sum()):.5f} | {np.mean(mm):.5f}')
f = (sid == 0) | (sid == 1); sds = np.mean([np.std(EA[i][f] - EA[jj][f]) for i, jj in ((0, 1), (0, 2), (1, 2))])
print(f'  W0S+W0L {int(f.sum())} feeds: implied SE {sds / np.sqrt(2) / np.sqrt(f.sum()):.5f}')
# same statistic on the stored holdout (GPU files), cross-user feeds, and on val
Eh = [ndcg_feeds(np.load(W5P / f'holdout/nn_w5_e10_xf0_s{s}_holdout.npy'), yh, oh, Mh) for s in S3]; cr = tm['xf'][hold] != 0
for nm, msk in (('holdout cross-user', cr), ('holdout fold-0', ~cr), ('holdout all', np.ones(len(Mh), bool))):
    sds = np.mean([np.std(Eh[i][msk] - Eh[jj][msk]) for i, jj in ((0, 1), (0, 2), (1, 2))])
    print(f'  {nm:18s} {int(msk.sum()):6d} feeds (stored GPU files): sd {sds:.4f} | implied SE {sds / np.sqrt(2) / np.sqrt(msk.sum()):.5f}')
ov = fv['cand_off'].to_numpy().astype(np.int64)
Ev = [ndcg_feeds(np.load(W5P / f'nn_w5_e10_xf0_s{s}_trainonly_val.npy'), yv, ov, Mv) for s in S3]
for nm, msk in (('val all', np.ones(len(Mv), bool)), ('val M<50', Mv < 50)):
    sds = np.mean([np.std(Ev[i][msk] - Ev[jj][msk]) for i, jj in ((0, 1), (0, 2), (1, 2))])
    print(f'  {nm:18s} {int(msk.sum()):6d} feeds (stored GPU files): sd {sds:.4f} | implied SE {sds / np.sqrt(2) / np.sqrt(msk.sum()):.5f}')
print('DONE')
