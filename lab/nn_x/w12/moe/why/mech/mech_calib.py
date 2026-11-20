"""w12/moe/why/mech: calibrators for the ablation numbers (CPU, stored predictions only).
 1. noise curve: NDCG change when i.i.d. Gaussian noise of within-feed relative size r is added to a reference member's logits
    (noise sd = r * pooled within-feed sd of the logits; 2 replicates; paired per feed).
 2. function distance between members: relative within-feed difference sqrt(sum (a - b)^2 / sum a^2) after within-feed centring and
    scaling each member to unit pooled within-feed sd, plus mean within-feed Pearson correlation, for ref-ref (different seeds),
    arm-ref (same seed = same data order and shared init of the common parameters), arm-ref (different seeds), arm-arm.
"""
import itertools
import json
import os
import sys
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/lab/nn_x/w12/moe/why/mech')
from mech_common import OUT, W5, FLAT, ndcg_fast, pred_paths, se  # noqa: E402
import polars as pl  # noqa: E402

f = pl.read_parquet(FLAT / 'val' / 'feeds.parquet')
Mv = f['M'].to_numpy().astype(np.int64); ov = np.concatenate([[0], np.cumsum(Mv)[:-1]])
yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32)
Mh, oh = np.load(W5 / 'preds/holdout/holdout_meta.npy'); yh = np.load(W5 / 'preds/holdout/holdout_y.npy').astype(np.float32)
SP = dict(val=(Mv, ov, yv), hold=(Mh.astype(np.int64), oh.astype(np.int64), yh))
ARMS = dict(ref='kd05w192', A='kd05w192moeffn', B='kd05w192moevo', C='kd05w192b3ff1')
SEEDS = (951, 952, 953, 954)


def load(arm, s, sp):
    pv, ph = pred_paths(f'w5_e10_{ARMS[arm]}_s{s}')
    return np.load(pv if sp == 'val' else ph).astype(np.float64)


def center_scale(x, M):
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, x, len(M)) / M
    c = x - mu[fid]
    return c / np.sqrt((c ** 2).mean()), fid


def wf_corr(a, b, fid, n):
    sab = np.bincount(fid, a * b, n); saa = np.bincount(fid, a * a, n); sbb = np.bincount(fid, b * b, n)
    return float(np.nanmean(sab / np.sqrt(saa * sbb)))


res = {'noise': {}, 'dist': {}}
rng = np.random.default_rng(7)
for sp, (M, o, y) in SP.items():
    for s in SEEDS:
        x = load('ref', s, sp)
        c, fid = center_scale(x, M)
        sd = np.sqrt(((x - (x - c * 0)) ** 2).mean())  # placeholder (unused)
        base = ndcg_fast(x.astype(np.float32), y, o, M)
        xc_sd = np.sqrt(((x - (np.bincount(fid, x, len(M)) / M)[fid]) ** 2).mean())
        for r in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
            ds = []
            for rep in range(2):
                xn = x + r * xc_sd * rng.standard_normal(len(x))
                ds.append(ndcg_fast(xn.astype(np.float32), y, o, M) - base)
            d = np.mean(ds, 0)
            res['noise'].setdefault(f'{sp}.r{r}', []).append((float(d.mean()), se(d)))
    for r in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
        v = res['noise'][f'{sp}.r{r}']
        print(f'noise {sp} r={r:.2f}: dNDCG {np.mean([a for a, _ in v]):+.5f} (per-seed SE ~{np.mean([b for _, b in v]):.5f})', flush=True)
    # function distances
    Z = {(a, s): center_scale(load(a, s, sp), M)[0] for a in ARMS for s in SEEDS}
    fid = np.repeat(np.arange(len(M)), M)
    def dist(p, q):
        return float(np.sqrt(((Z[p] - Z[q]) ** 2).mean())), wf_corr(Z[p], Z[q], fid, len(M))
    rows = {}
    rows['ref-ref (diff seed)'] = [dist(('ref', a), ('ref', b)) for a, b in itertools.combinations(SEEDS, 2)]
    for arm in ('A', 'B', 'C'):
        rows[f'{arm}-ref (same seed)'] = [dist((arm, s), ('ref', s)) for s in SEEDS]
        rows[f'{arm}-ref (diff seed)'] = [dist((arm, a), ('ref', b)) for a in SEEDS for b in SEEDS if a != b]
        rows[f'{arm}-{arm} (diff seed)'] = [dist((arm, a), (arm, b)) for a, b in itertools.combinations(SEEDS, 2)]
    # pool-level: 4-pool of arm vs 4-pool of ref (mean of the scaled members)
    pool = {a: np.mean([Z[(a, s)] for s in SEEDS], 0) for a in ARMS}
    for a in ARMS:
        pool[a] = pool[a] / np.sqrt((pool[a] ** 2).mean())
    for arm in ('A', 'B', 'C'):
        rows[f'{arm} 4-pool vs ref 4-pool'] = [(float(np.sqrt(((pool[arm] - pool['ref']) ** 2).mean())), wf_corr(pool[arm], pool['ref'], fid, len(M)))]
    # split-half ref pools (951+952 vs 953+954) for scale
    h1 = Z[('ref', 951)] + Z[('ref', 952)]; h2 = Z[('ref', 953)] + Z[('ref', 954)]
    h1 /= np.sqrt((h1 ** 2).mean()); h2 /= np.sqrt((h2 ** 2).mean())
    rows['ref 2-pool vs ref 2-pool (disjoint seeds)'] = [(float(np.sqrt(((h1 - h2) ** 2).mean())), wf_corr(h1, h2, fid, len(M)))]
    for k, v in rows.items():
        res['dist'][f'{sp}.{k}'] = dict(reldiff=float(np.mean([a for a, _ in v])), corr=float(np.mean([b for _, b in v])), n=len(v))
        print(f'dist {sp} {k:42s}: rel diff {np.mean([a for a, _ in v]):.3f}  within-feed corr {np.mean([b for _, b in v]):.4f}  (n pairs {len(v)})', flush=True)
(OUT / 'calib.json').write_text(json.dumps(res, indent=1))
