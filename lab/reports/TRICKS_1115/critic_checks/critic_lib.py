"""Shared helpers of the critic's checks (own code; CPU; read-only on project files). Only src/pmix.weights() is reused (definition of the feed mixes)."""
import sys
sys.path.insert(0, '<repo>/src')
import numpy as np, polars as pl
from pathlib import Path
import pmix
R = Path('<repo>'); P = R / 'preds'; FLAT = R / 'data/flat'; OUT = R / 'reports/TRICKS_1115/critic_checks'
fv = pl.read_parquet(FLAT / 'val/feeds.parquet'); Mv = fv['M'].to_numpy().astype(np.int64); ov = fv['cand_off'].to_numpy().astype(np.int64)
ft = pl.read_parquet(FLAT / 'test/feeds.parquet'); Mt = ft['M'].to_numpy().astype(np.int64); ot = ft['cand_off'].to_numpy().astype(np.int64)
yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64); UID = fv['user_id'].to_numpy()
FID = {'val': np.repeat(np.arange(len(Mv)), Mv), 'test': np.repeat(np.arange(len(Mt)), Mt)}; MM = {'val': Mv, 'test': Mt}
NV = len(Mv); NPUB = 8562; NPRIV = 20033
def z(x, sp='val'):
    M = MM[sp]; fid = FID[sp]; x = np.asarray(x, np.float32).astype(np.float64)
    mu = np.bincount(fid, x, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, len(M)) / M - mu * mu, 0.0)) + 1e-9
    return (x - mu[fid]) / sd[fid]
def ndcg(s):
    s = np.asarray(s, np.float32).astype(np.float64); fid = FID['val']
    o = np.lexsort((yv, -s, fid)); pos = np.arange(len(s)) - ov[fid]
    d = 1.0 / np.log2(pos + 2.0)
    dcg = np.bincount(fid, yv[o] * d, NV); npos = np.bincount(fid, yv, NV); idcg = np.bincount(fid, (pos < npos[fid]) * d, NV)
    out = np.ones(NV); m = idcg > 0; out[m] = dcg[m] / idcg[m]; return out
W = pmix.weights(); WPUB, WPRIV = W['public'], W['private']
def wm(d, w=None):
    if w is None: return float(d.mean()), float(d.std() / np.sqrt(len(d)))
    m = np.average(d, weights=w); ne = w.sum() ** 2 / (w ** 2).sum(); return float(m), float(np.sqrt(np.average((d - m) ** 2, weights=w) / ne))
f3 = lambda t: f'{t[0]:+.5f} +- {t[1]:.5f}'
def three(d): return f'val {f3(wm(d))} | pmix-public {f3(wm(d, WPUB))} | pmix-private {f3(wm(d, WPRIV))}'
SUF = {'val': ('n2cpv3s3_val_oof', 'trainonly_val'), 'test': ('n2cpv3s3_test', 'trainonly_test')}
_ZC = {}
def mz(m, sp='val'):
    """(z of the fine-tuned prediction, z of the train-only prediction) of one member, cached"""
    k = (m, sp)
    if k not in _ZC: _ZC[k] = (z(np.load(P / f'nn_{m}_{SUF[sp][0]}.npy'), sp), z(np.load(P / f'nn_{m}_{SUF[sp][1]}.npy'), sp))
    return _ZC[k]
def pool(names, sp='val'):
    a = np.zeros(len(FID[sp])); b = np.zeros(len(FID[sp]))
    for m in names:
        u, v = mz(m, sp); a += u; b += v
    return a / len(names), b / len(names)
def branches(base, pop=(), w=0.2, sp='val'):
    b = pool(base, sp)
    if len(pop):
        p = pool(pop, sp); b = ((1 - w) * z(b[0], sp) + w * z(p[0], sp), (1 - w) * z(b[1], sp) + w * z(p[1], sp))
    return b
def fin(br, sp='val', wft=0.5): return wft * z(br[0], sp) + (1 - wft) * z(br[1], sp)
def stored(tag, sp='val', d=P, wft=0.5): return wft * z(np.load(d / f'ens_{tag}_ft_{sp}.npy'), sp) + (1 - wft) * z(np.load(d / f'ens_{tag}_to_{sp}.npy'), sp)
S8 = list(range(951, 959)); S4 = list(range(951, 955)); S4b = list(range(955, 959))
wide = lambda pre, seeds: [f'w5_{pre}_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in seeds]
def user_folds(rng, k=5):
    """user-grouped folds; every user-0 feed is its own group (pooled bucket)"""
    g = UID.astype(np.int64).copy(); z0 = np.where(g == 0)[0]; g[z0] = -1 - np.arange(len(z0))
    u, inv = np.unique(g, return_inverse=True); return rng.integers(0, k, len(u))[inv]
