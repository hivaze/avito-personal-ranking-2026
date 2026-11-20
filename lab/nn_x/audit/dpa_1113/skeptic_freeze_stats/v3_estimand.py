"""Estimand check: the proposer reads VAL-fitted family weights (fitted on the 0.5 FT + 0.5 TO blend) on the HOLDOUT in the
TO-only form. Is the val->holdout disagreement 'val noise' or an FT-vs-TO / M-mix estimand mismatch?
Fit family weights on the val TO half only (same form as the holdout), on all feeds and on M>=50 only; read on the holdout."""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, time
sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from metrics import ndcg_per_feed
from common import feed_meta, FLAT
import pmix

T0 = time.time()
ROOT = Path('<repo>'); P = ROOT / 'preds'; HD = P / 'holdout'
OUT = Path('/tmp/scratchpad/dpa/skeptic_freeze_stats')
log = open(OUT / 'v3_estimand.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
def fz(x, off, M):
    x = x.astype(np.float64); fid = np.repeat(np.arange(len(M)), M)
    mu = np.add.reduceat(x, off) / M; d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, off) / M) + 1e-9
    return d / sd[fid]
def se_user(d, g):
    n = len(d); mu = d.mean(); iid = d.std() / np.sqrt(n)
    cl = np.unique(g, return_inverse=True)[1]; s = np.bincount(cl, d - mu); k = (s != 0).sum()
    return mu, max(iid, np.sqrt((s ** 2).sum()) / n * np.sqrt(k / max(k - 1, 1)))
FAM = {'F10': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)], 'F7x': [f'r6_cd_m02_r{s}' for s in range(101, 108)],
       'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]}
Mh, offh = np.load(HD / 'holdout_meta.npy'); Mh = Mh.astype(np.int64); offh = offh.astype(np.int64)
yh = np.load(HD / 'holdout_y.npy').astype(np.float64); uh = np.load(ROOT / 'nn_x/stack/nnstack/data/hold_user.npy')
gh = np.where(uh == 0, -1 - np.arange(len(Mh)), uh)
fv, Mv, offv = feed_meta('val'); Mv = Mv.astype(np.int64); offv = offv.astype(np.int64)
yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64); uv = fv['user_id'].to_numpy(); gv = np.where(uv == 0, -1 - np.arange(len(Mv)), uv)
Fv = [np.mean([fz(np.load(P / f'nn_{m}_trainonly_val.npy'), offv, Mv) for m in FAM[f]], 0) for f in FAM]
NH = np.load(OUT / 'v1_NH.npy').astype(np.float64); A = np.load(OUT / 'v1_A.npy'); NVb = np.load(OUT / 'v1_NV.npy').astype(np.float64)
NVt = np.stack([ndcg_per_feed(a[0] * Fv[0] + a[1] * Fv[1] + a[2] * Fv[2], yv, offv, Mv) for a in A], 1)
pr(f'[{time.time()-T0:.0f}s] val TO pool (ME) {NVt[:, 0].mean():.5f}')
m50 = Mv >= 50
mh = NH.mean(0)
for nm, S in [('val TO all', NVt.mean(0)), ('val TO M>=50', NVt[m50].mean(0)), ('val TO M<50', NVt[~m50].mean(0)),
              ('val blend all', NVb.mean(0)), ('val blend M>=50', NVb[m50].mean(0))]:
    i = 1 + int(np.argmax(S[1:]))
    dh = NH[:, i] - NH[:, 0]; h, hs = se_user(dh, gh)
    pr(f'{nm:16s} corr with holdout surface {np.corrcoef(S, mh)[0, 1]:+.3f} | val-fit a={tuple(np.round(A[i], 2))} in-sample {S[i] - S[0]:+.5f} '
       f'-> HOLD {h:+.5f} +- {hs:.5f} (z {h / hs:+.2f})')
# repeated nested TO-form fit on val (comparable to the holdout-fit OOF)
ug = np.unique(gv, return_inverse=True)[1]; nU = ug.max() + 1; R = []
for rep in range(30):
    fold = np.random.default_rng(4000 + rep).integers(0, 5, nU)[ug]; oof = np.zeros(len(Mv))
    for k in range(5):
        tr = fold != k; i = 1 + int(np.argmax(NVt[tr, 1:].mean(0))); oof[fold == k] = NVt[fold == k, i]
    R.append(se_user(oof - NVt[:, 0], gv))
R = np.array(R)
pr(f'val TO-form nested family-weight fit (30 splits): mean {R[:, 0].mean():+.5f}, median z {np.median(R[:, 0] / R[:, 1]):+.2f}, share>0 {(R[:, 0] > 0).mean():.2f}')
pr(f'[{time.time()-T0:.0f}s] done')
