"""Skeptic check of the one combiner arm the freeze closes that PASSED val nested CV in the proposer's own run
(e23_grid 'M2 a1_wft': w_ft 0.3 for M<50 / 0.6 for M>=50, nested z +3.15, vs-uncond z +2.24).
(1) repeat the nested fit over 30 random user-grouped 5-fold splits; (2) placebo: same-size random feed buckets;
(3) frozen rule on the two newest independent 10-epoch families (popfree, pop1) and on G3 halves by family (F10 / F7x / F7x70)."""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, time, glob, re
sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from metrics import ndcg_per_feed
from common import feed_meta, FLAT
import pmix

T0 = time.time()
ROOT = Path('<repo>'); P = ROOT / 'preds'
OUT = Path('/tmp/scratchpad/dpa/skeptic_freeze_stats')
log = open(OUT / 'v2_wft_mcond.txt', 'w')
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

fv, M, off = feed_meta('val'); M = M.astype(np.int64); off = off.astype(np.int64)
y = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
u = fv['user_id'].to_numpy(); g = np.where(u == 0, -1 - np.arange(len(M)), u)
ug = np.unique(g, return_inverse=True)[1]; nU = ug.max() + 1
wpriv = pmix.weights()['private']
G3 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)] + [f'r6_cd_m02_r{s}' for s in range(101, 108)] + [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]
WFT = np.array([0.3, 0.4, 0.5, 0.6, 0.7]); i5 = 2

def curves(ms):
    zf = fz(np.mean([fz(np.load(P / f'nn_{m}_n2cpv3_val_oof.npy'), off, M) for m in ms], 0), off, M)
    zt = fz(np.mean([fz(np.load(P / f'nn_{m}_trainonly_val.npy'), off, M) for m in ms], 0), off, M)
    return np.stack([ndcg_per_feed(w * zf + (1 - w) * zt, y, off, M) for w in WFT], 1)

NV = curves(G3)
pr(f'[{time.time()-T0:.0f}s] G3 w.5 val {NV[:, i5].mean():.5f}')
b = (M >= 50).astype(int)

def nested(bk, fold):
    oofc = np.zeros(len(M)); oofu = np.zeros(len(M))
    for k in range(5):
        tr = fold != k; te = fold == k
        oofu[te] = NV[te, int(np.argmax(NV[tr].mean(0)))]
        for v in np.unique(bk):
            j = int(np.argmax(NV[tr & (bk == v)].mean(0))); oofc[te & (bk == v)] = NV[te & (bk == v), j]
    return oofc, oofu

R = []; PL = []
for rep in range(30):
    fold = np.random.default_rng(3000 + rep).integers(0, 5, nU)[ug]
    oofc, oofu = nested(b, fold)
    a, sa = se_user(oofc - NV[:, i5], g); c, sc = se_user(oofc - oofu, g)
    R.append((a, a / sa, c / sc, np.average(oofc - NV[:, i5], weights=wpriv)))
    # placebo: random bucket per feed with the same share as M>=50
    pb = (np.random.default_rng(5000 + rep).random(len(M)) < b.mean()).astype(int)
    oofp, _ = nested(pb, fold); p, sp = se_user(oofp - NV[:, i5], g); PL.append((p, p / sp))
R = np.array(R); PL = np.array(PL)
pr(f'M2 a1_wft nested, 30 random user-grouped splits: delta vs w.5 mean {R[:, 0].mean():+.5f} (min {R[:, 0].min():+.5f} max {R[:, 0].max():+.5f}), '
   f'median z {np.median(R[:, 1]):+.2f}, share z>2 {(R[:, 1] > 2).mean():.2f}; vs uncond median z {np.median(R[:, 2]):+.2f}; mean pmix {R[:, 3].mean():+.5f}')
pr(f'  placebo (random same-size buckets): mean {PL[:, 0].mean():+.5f}, median z {np.median(PL[:, 1]):+.2f}, max z {PL[:, 1].max():+.2f}')

# frozen rule (0.3 | 0.6) on families not in G3 and on G3's own families separately
w = np.where(np.repeat(M, M) < 50, 0.3, 0.6)
def rule(ms, tag):
    zf = fz(np.mean([fz(np.load(P / f'nn_{m}_n2cpv3_val_oof.npy'), off, M) for m in ms], 0), off, M)
    zt = fz(np.mean([fz(np.load(P / f'nn_{m}_trainonly_val.npy'), off, M) for m in ms], 0), off, M)
    d = ndcg_per_feed(w * zf + (1 - w) * zt, y, off, M) - ndcg_per_feed(0.5 * zf + 0.5 * zt, y, off, M)
    mu, se = se_user(d, g); a, sa = se_user(d[M < 50], g[M < 50]); c, sc = se_user(d[M >= 50], g[M >= 50])
    pr(f'  frozen rule {tag:28s} n={len(ms):2d}: {mu:+.5f} +- {se:.5f} (z {mu / se:+.2f}) pmix {np.average(d, weights=wpriv):+.5f} | M<50 {a:+.5f} (z {a / sa:+.2f}) M>=50 {c:+.5f} (z {c / sc:+.2f})')
    return d
have = lambda m: (P / f'nn_{m}_n2cpv3_val_oof.npy').exists() and (P / f'nn_{m}_trainonly_val.npy').exists()
for pre in ['abl_e10_xdrop_pop', 'abl_e10_pop1']:
    ms = sorted({re.sub(r'^nn_|_trainonly_val\.npy$', '', Path(f).name) for f in glob.glob(str(P / f'nn_{pre}_s*_trainonly_val.npy'))})
    ms = [m for m in ms if not m.endswith('np') and '@' not in m and have(m)]
    if len(ms) >= 3: rule(ms, pre + ' (fresh)')
    else: pr(pre, 'only', ms)
for nm, ms in [('F10 (in G3)', G3[:6]), ('F7x (in G3)', G3[6:13]), ('F7x70 (in G3)', G3[13:])]:
    rule(ms, nm)
pr(f'[{time.time()-T0:.0f}s] done')
