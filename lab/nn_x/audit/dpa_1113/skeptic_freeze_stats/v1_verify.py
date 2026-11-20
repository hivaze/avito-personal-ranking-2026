"""Skeptic (statistical validity) re-check of the FREEZE-ME proposal. Independent of the proposer's cache:
raw preds -> own per-feed z -> family means -> official src/metrics.ndcg_per_feed. CPU only. Read-only on project files."""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, time
sys.path.insert(0, '<repo>/src')
import numpy as np
import polars as pl
from pathlib import Path
from metrics import ndcg_per_feed
from common import feed_meta, FLAT
import pmix

T0 = time.time()
ROOT = Path('<repo>'); P = ROOT / 'preds'; HD = P / 'holdout'
OUT = Path('/tmp/scratchpad/dpa/skeptic_freeze_stats')
log = open(OUT / 'v1_verify.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

FAM = {'F10': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)],
       'F7x': [f'r6_cd_m02_r{s}' for s in range(101, 108)],
       'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]}

def fz(x, off, M):
    x = x.astype(np.float64); fid = np.repeat(np.arange(len(M)), M)
    mu = np.add.reduceat(x, off) / M
    d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, off) / M) + 1e-9
    return d / sd[fid]

def se_user(d, g):
    """max(iid, user-clustered) SE of mean(d)."""
    n = len(d); mu = d.mean(); iid = d.std() / np.sqrt(n)
    cl = np.unique(g, return_inverse=True)[1]; s = np.bincount(cl, d - mu); k = (s != 0).sum()
    return mu, max(iid, np.sqrt((s ** 2).sum()) / n * np.sqrt(k / max(k - 1, 1)))

# ---------------- holdout ----------------
Mh, offh = np.load(HD / 'holdout_meta.npy'); Mh = Mh.astype(np.int64); offh = offh.astype(np.int64)
yh = np.load(HD / 'holdout_y.npy').astype(np.float64)
uh = np.load(ROOT / 'nn_x/stack/nnstack/data/hold_user.npy')
gh = np.where(uh == 0, -1 - np.arange(len(Mh)), uh)
Fh = [np.mean([fz(np.load(HD / f'nn_{m}_holdout.npy'), offh, Mh) for m in FAM[f]], 0) for f in FAM]
# ---------------- val ----------------
fv, Mv, offv = feed_meta('val'); Mv = Mv.astype(np.int64); offv = offv.astype(np.int64)
yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
uv = fv['user_id'].to_numpy(); gv = np.where(uv == 0, -1 - np.arange(len(Mv)), uv)
Fv = {h: [np.mean([fz(np.load(P / pat.format(m)), offv, Mv) for m in FAM[f]], 0) for f in FAM]
      for h, pat in [('ft', 'nn_{}_n2cpv3_val_oof.npy'), ('to', 'nn_{}_trainonly_val.npy')]}
wpriv = pmix.weights()['private']; wpub = pmix.weights()['public']
pr(f'[{time.time()-T0:.0f}s] loaded; holdout feeds {len(Mh)} (M<50: {(Mh < 50).sum()}), val feeds {len(Mv)} (M<50 {(Mv < 50).mean():.3f})')

# grid: step 0.1 simplex (66) + exact ME
A = [(i / 10, j / 10, round(1 - (i + j) / 10, 10)) for i in range(11) for j in range(11) if i + j <= 10]
ME = (6 / 20, 7 / 20, 7 / 20); A = np.array([ME] + A); iME = 0
NH = np.zeros((len(Mh), len(A))); NV = np.zeros((len(Mv), len(A)))
for ia, a in enumerate(A):
    NH[:, ia] = ndcg_per_feed(a[0] * Fh[0] + a[1] * Fh[1] + a[2] * Fh[2], yh, offh, Mh)
    zf = fz(a[0] * Fv['ft'][0] + a[1] * Fv['ft'][1] + a[2] * Fv['ft'][2], offv, Mv)
    zt = fz(a[0] * Fv['to'][0] + a[1] * Fv['to'][1] + a[2] * Fv['to'][2], offv, Mv)
    NV[:, ia] = ndcg_per_feed(0.5 * zf + 0.5 * zt, yv, offv, Mv)
pr(f'[{time.time()-T0:.0f}s] ME: holdout {NH[:, iME].mean():.5f} (rec. 0.32379) val {NV[:, iME].mean():.5f} (rec. 0.40032)')
np.save(OUT / 'v1_NH.npy', NH.astype(np.float32)); np.save(OUT / 'v1_NV.npy', NV.astype(np.float32)); np.save(OUT / 'v1_A.npy', A)

mh = NH.mean(0); mv = NV.mean(0)
mvw = (NV * wpriv[:, None]).sum(0) / wpriv.sum(); mvpub = (NV * wpub[:, None]).sum(0) / wpub.sum()
m50 = Mv >= 50
mv50 = NV[m50].mean(0); mvlt = NV[~m50].mean(0)
def best(m): i = int(np.argmax(m)); return f'a={tuple(np.round(A[i], 2))} {m[i] - m[iME]:+.5f}'
pr('SURFACE argmax (vs ME) holdout', best(mh), '| val', best(mv), '| val pmix(priv)', best(mvw), '| val pub-w', best(mvpub),
   '| val M>=50', best(mv50), '| val M<50', best(mvlt))
for nm, s in [('val all', mv), ('val pmix', mvw), ('val M>=50', mv50), ('val M<50', mvlt)]:
    pr(f'  corr(holdout surface, {nm} surface) over {len(A)} pts: {np.corrcoef(mh, s)[0, 1]:+.3f}')
# key closed points, holdout and val (+ val M>=50 slice) deltas with SE
for nm, a in [('F10 0', (0, .5, .5)), ('val-fit (0.05,.35,.6)~(0,.4,.6)', (0, .4, .6)), ('(0.1,.4,.5)', (.1, .4, .5)), ('hold-best (.3,.4,.3)', (.3, .4, .3))]:
    k = int(np.argmin(np.abs(A[1:] - np.array(a)).sum(1))) + 1
    dh = NH[:, k] - NH[:, iME]; dv = NV[:, k] - NV[:, iME]
    h, hs = se_user(dh, gh); v, vs = se_user(dv, gv); v5, v5s = se_user(dv[m50], gv[m50]); vl, vls = se_user(dv[~m50], gv[~m50])
    pr(f'  {nm:34s} a={tuple(np.round(A[k], 2))}: HOLD {h:+.5f}+-{hs:.5f} (z {h / hs:+.2f}) | VAL {v:+.5f} (z {v / vs:+.2f}) | '
       f'VAL M>=50 {v5:+.5f} (z {v5 / v5s:+.2f}) | VAL M<50 {vl:+.5f} (z {vl / vls:+.2f}) | pmix {np.average(dv, weights=wpriv):+.5f}')

# ---------------- repeated user-grouped holdout-fit OOF (proposer used ONE GroupKFold split) ----------------
ug = np.unique(gh, return_inverse=True)[1]; nU = ug.max() + 1
res = []
for rep in range(30):
    fold = np.random.default_rng(1000 + rep).integers(0, 5, nU)[ug]
    oof = np.zeros(len(Mh))
    for k in range(5):
        tr = fold != k; ia = 1 + int(np.argmax(NH[tr, 1:].mean(0))); oof[fold == k] = NH[fold == k, ia]
    d = oof - NH[:, iME]; mu, se = se_user(d, gh); res.append((mu, se))
res = np.array(res)
pr(f'holdout-fit OOF (step .1 grid, 30 random user-grouped 5-fold splits): mean delta {res[:, 0].mean():+.5f} '
   f'(min {res[:, 0].min():+.5f} max {res[:, 0].max():+.5f}); share of splits with delta > 0: {(res[:, 0] > 0).mean():.2f}; '
   f'median z {np.median(res[:, 0] / res[:, 1]):+.2f}')
# same with ME allowed in the grid (a refit that can return ME exactly)
res2 = []
for rep in range(30):
    fold = np.random.default_rng(1000 + rep).integers(0, 5, nU)[ug]
    oof = np.zeros(len(Mh))
    for k in range(5):
        tr = fold != k; ia = int(np.argmax(NH[tr].mean(0))); oof[fold == k] = NH[fold == k, ia]
    d = oof - NH[:, iME]; res2.append(se_user(d, gh))
res2 = np.array(res2)
pr(f'  same incl. exact ME in the grid: mean {res2[:, 0].mean():+.5f}, share > 0 {(res2[:, 0] > 0).mean():.2f}, median z {np.median(res2[:, 0] / res2[:, 1]):+.2f}')

# ---------------- val-fit family weights: repeated user-grouped nested OOF on val ----------------
ugv = np.unique(gv, return_inverse=True)[1]; nUv = ugv.max() + 1
rv = []
for rep in range(30):
    fold = np.random.default_rng(2000 + rep).integers(0, 5, nUv)[ugv]
    oof = np.zeros(len(Mv))
    for k in range(5):
        tr = fold != k; ia = 1 + int(np.argmax(NV[tr, 1:].mean(0))); oof[fold == k] = NV[fold == k, ia]
    d = oof - NV[:, iME]; mu, se = se_user(d, gv); rv.append((mu, se, np.average(d, weights=wpriv)))
rv = np.array(rv)
pr(f'val-fit family weights nested OOF (step .1, 30 random splits): mean {rv[:, 0].mean():+.5f} (min {rv[:, 0].min():+.5f} max {rv[:, 0].max():+.5f}), '
   f'median z {np.median(rv[:, 0] / rv[:, 1]):+.2f}, share z>2 {(rv[:, 0] / rv[:, 1] > 2).mean():.2f}, mean pmix {rv[:, 2].mean():+.5f}')

# ---------------- test mix (representativeness of the holdout as the arbiter) ----------------
ft_, Mt, offt = feed_meta('test'); tst = ft_['timestamp'].to_numpy(); pub = tst < 1790497920
pr(f'M<50 share: val {(Mv < 50).mean():.3f} | holdout {(Mh < 50).mean():.3f} | test public {(Mt[pub] < 50).mean():.3f} private {(Mt[~pub] < 50).mean():.3f}')
pr(f'mean M: val {Mv.mean():.1f} holdout {Mh.mean():.1f} public {Mt[pub].mean():.1f} private {Mt[~pub].mean():.1f}')
pr(f'[{time.time()-T0:.0f}s] done')
