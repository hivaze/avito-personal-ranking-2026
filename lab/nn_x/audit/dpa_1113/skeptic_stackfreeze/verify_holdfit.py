"""Independent re-computation (skeptic) of the FREEZE-ME key numbers on the holdout, built straight from preds/holdout
(not from the proposer's cache): family-weight simplex (step .05) on the TO holdout pool, user-grouped 5-fold OOF of the
holdout-fitted weights vs member-equal, and the val-fitted weights a=(0.05,0.35,0.6) applied to the holdout.
Read-only on project files. CPU only."""
import os
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS']:
    os.environ[k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, time
sys.path.insert(0, '<repo>/src')
import numpy as np
from metrics import ndcg_per_feed
from sklearn.model_selection import GroupKFold

T0 = time.time()
R = '<repo>/'
M, off = np.load(R + 'preds/holdout/holdout_meta.npy').astype(np.int64)
y = np.load(R + 'preds/holdout/holdout_y.npy').astype(np.float64)
u = np.load(R + 'nn_x/stack/nnstack/data/hold_user.npy')
n = len(M); fid = np.repeat(np.arange(n), M)
FAM = {'F10': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)],
       'F7x': [f'r6_cd_m02_r{s}' for s in range(101, 108)],
       'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]}


def fz(x):
    x = x.astype(np.float64)
    mu = np.add.reduceat(x, off) / M
    d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, off) / M) + 1e-9
    return d / sd[fid]


famz = []
for f, ms in FAM.items():
    famz.append(np.mean([fz(np.load(R + f'preds/holdout/nn_{m}_holdout.npy')) for m in ms], 0))
famz = np.stack(famz)
nmem = np.array([len(v) for v in FAM.values()], float)
me = nmem / nmem.sum()
base = ndcg_per_feed(me @ famz, y, off, M)
print(f'[{time.time()-T0:.0f}s] ME holdout TO pool NDCG {base.mean():.6f}', flush=True)

g = np.where(u == 0, -1 - np.arange(n), u)
cl = np.unique(g, return_inverse=True)[1]


def se_cl(d):
    mu = d.mean(); s = np.bincount(cl, d - mu)
    G = (s != 0).sum()
    return mu, max(d.std() / np.sqrt(len(d)), np.sqrt((s ** 2).sum()) / len(d) * np.sqrt(G / (G - 1)))


A = []
for i in range(21):
    for j in range(21 - i):
        A.append((i / 20, j / 20, (20 - i - j) / 20))
A = np.array(A)
NH = np.zeros((n, len(A)), np.float32)
for k, a in enumerate(A):
    NH[:, k] = ndcg_per_feed(a @ famz, y, off, M)
np.save(os.path.dirname(__file__) + '/NH_grid.npy', NH)
print(f'[{time.time()-T0:.0f}s] grid done', flush=True)
mean = NH.mean(0); ib = int(mean.argmax())
print(f'holdout surface: best a={A[ib]} {mean[ib]:.6f} vs ME {base.mean():.6f} -> in-sample +{mean[ib]-base.mean():.5f}')
iv = int(np.argmin(np.abs(A - np.array([0.05, 0.35, 0.6])).sum(1)))
d = NH[:, iv] - base; m_, s_ = se_cl(d)
print(f'val-fitted a={A[iv]} on holdout: {m_:+.5f} +- {s_:.5f} (z {m_/s_:+.2f})')


def oof(folds, allowed):
    o = np.zeros(n); ch = []
    for k in range(folds.max() + 1):
        tr = folds != k
        ia = max(allowed, key=lambda j: NH[tr, j].mean())
        o[folds == k] = NH[folds == k, ia]; ch.append(tuple(A[ia]))
    return se_cl(o - base), ch


hf = np.zeros(n, int)
for k, (_, te) in enumerate(GroupKFold(5).split(np.arange(n), groups=g)):
    hf[te] = k
coarse = [i for i in range(len(A)) if np.allclose(np.round(A[i] * 10), A[i] * 10)]
for nm, al in [('step .05', list(range(len(A)))), ('step .1', coarse)]:
    (m_, s_), ch = oof(hf, al)
    print(f'GroupKFold OOF holdout-fit ({nm}): {m_:+.5f} +- {s_:.5f} (z {m_/s_:+.2f}) choices {ch}')
# robustness: 10 random user-hash 5-fold splits
rs = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    perm = rng.integers(0, 5, cl.max() + 1)
    (m_, s_), ch = oof(perm[cl], list(range(len(A))))
    rs.append(m_)
    print(f'  random user split seed {seed}: OOF {m_:+.5f} +- {s_:.5f} (z {m_/s_:+.2f})')
print(f'random-split OOF mean {np.mean(rs):+.5f}, min {np.min(rs):+.5f}, max {np.max(rs):+.5f}, n>0: {sum(r > 0 for r in rs)}/10')
print(f'[{time.time()-T0:.0f}s] done')
