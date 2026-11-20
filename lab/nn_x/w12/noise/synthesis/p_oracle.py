"""Examination oracle on test-like (random-order) train feeds, split into slot examination and the last displayed item.
Reads the member scores saved by p_token.py (row-1 = random-order domain row, the row used on val/test) and the flat arrays.
The oracle adds log e(bin) to the score; e is estimated on the OTHER random-order feeds of the same group (disjoint).
Also: production feeds, click mass by display-rank bin with and without the last item (actual vs model row0 vs row1).
Usage: python p_oracle.py <checkpoint name>   -> out/p_oracle_<ckpt>.txt
"""
import sys
from pathlib import Path
import numpy as np, polars as pl

sys.path.insert(0, '<repo>')
from avito_ranker.metrics import ndcg_per_feed

ROOT = Path('<repo>'); OUT = ROOT / 'lab/nn_x/w12/noise/synthesis/out'; FLAT = ROOT / 'artifacts/data/flat/train'
name = sys.argv[1]
f = pl.read_parquet(FLAT / 'feeds.parquet'); M = f['M'].to_numpy().astype(np.int64); off = f['cand_off'].to_numpy().astype(np.int64)
tgt = np.load(FLAT / 'cand_target.npy', mmap_mode='r'); pos = np.load(FLAT / 'cand_pos.npy', mmap_mode='r')
p_old = np.nan_to_num(np.load(ROOT / 'artifacts/exploration/train_prandom.npz')['p_random'].astype(np.float32), nan=-1)
p_v2 = np.load(ROOT / 'lab/eda/labels_position/out/train_prandom_v2.npz')['p_random'].astype(np.float32)
pools = {'rand_M50+': np.flatnonzero((M >= 50) & (p_old > 0.9) & (p_v2 > 0.9)), 'rand_M<50': np.flatnonzero((M < 50) & (p_v2 > 0.9)),
         'prod_M50+': np.flatnonzero((M >= 50) & (p_old >= 0) & (p_old < 0.05) & (p_v2 < 0.05)), 'prod_M<50': np.flatnonzero((M < 50) & (p_v2 < 0.05))}
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 199), (200, 10**6)]
NB = len(BINS)


def layout(fids):
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fids])
    fidx = np.repeat(np.arange(len(fids)), M[fids])
    order = np.lexsort((np.asarray(pos[rows]), fidx))
    start = np.concatenate([[0], np.cumsum(M[fids])[:-1]])
    rk = np.empty(len(rows), np.int64); rk[order] = np.arange(len(rows)) - np.repeat(start, M[fids])
    b = np.full(len(rows), -1)
    for k, (lo, hi) in enumerate(BINS):
        b[(rk >= lo) & (rk <= hi)] = k
    last = rk == np.repeat(M[fids], M[fids]) - 1
    return rows, b, last, start


lines = [f'checkpoint {name}; scores from out/p_token_{name}_<group>.npz (row1 = random-order domain row)']
for g in ('rand_M50+', 'rand_M<50'):
    z = np.load(OUT / f'p_token_{name}_{g}.npz'); ids = z['ids']; s1 = z['s1'].astype(np.float64)
    rows, b, last, start = layout(ids); y = np.asarray(tgt[rows], np.float64)
    ref = np.setdiff1d(pools[g], ids); rr, rb, rl, _ = layout(ref); ry = np.asarray(tgt[rr], np.float64)
    # within-feed relative examination: observed clicks / clicks expected if every non-last slot of the feed had the
    # feed's own non-last click rate (removes the confound between deep slots and long, low-rate feeds)
    _, rst = np.unique(np.repeat(np.arange(len(ref)), M[ref]), return_index=True)
    knl = np.add.reduceat(ry * ~rl, rst); feed_rate = knl / (M[ref] - 1)
    exp_r = np.repeat(feed_rate, M[ref])
    slot = np.array([ry[(rb == k) & ~rl].sum() / exp_r[(rb == k) & ~rl].sum() if ((rb == k) & ~rl).any() else np.nan for k in range(NB)])
    nonlast = ry[~rl].mean(); lastr = ry[rl].mean()
    lslot = np.log(np.where(np.isnan(slot), 1.0, slot))
    base = ndcg_per_feed(s1, y, start, M[ids])
    arms = {'slot examination only (last item scored by its rank bin)': s1 + lslot[b],
            'last displayed item only': s1 + np.where(last, np.log(lastr / nonlast), 0.0),
            'both': s1 + lslot[b] + np.where(last, np.log(lastr / nonlast), 0.0)}
    lines.append(f'\n== {g}: {len(ids)} scored feeds, mean M {M[ids].mean():.0f}; e from {len(ref):,} disjoint feeds')
    lines.append(f'  within-feed relative examination by slot bin (non-last rows): {np.round(slot, 3).tolist()}; non-last click rate {nonlast:.4f}; last displayed item {lastr:.4f}')
    lines.append(f'  share of clicks on the last displayed item: {ry[rl].sum()/ry.sum():.3f}')
    lines.append(f'  member NDCG (row1): {base.mean():.5f}')
    for a, s in arms.items():
        d = ndcg_per_feed(s, y, start, M[ids]) - base
        lines.append(f'  oracle {a}: {d.mean():+.5f} +- {d.std(ddof=1)/np.sqrt(len(d)):.5f}')

for g in ('prod_M50+', 'prod_M<50'):
    z = np.load(OUT / f'p_token_{name}_{g}.npz'); ids = z['ids']
    rows, b, last, start = layout(ids); y = np.asarray(tgt[rows], np.float64)
    K = np.add.reduceat(y, start)
    q = {}
    for r in ('s0', 's1'):
        s = z[r].astype(np.float64); fe = np.repeat(np.arange(len(ids)), M[ids])
        mx = np.maximum.reduceat(s, start); e = np.exp(s - mx[fe]); q[r] = e / np.add.reduceat(e, start)[fe] * K[fe]
    keep = ~last
    def dist(w):
        h = np.array([w[(b == k) & keep].sum() for k in range(NB)]); return h / h.sum()
    da, d0, d1 = dist(y), dist(q['s0']), dist(q['s1'])
    tv = lambda a, c: 0.5 * np.abs(a - c).sum()
    lines.append(f'\n== {g} (last displayed item excluded): click-mass TV over slot bins: actual vs row0 {tv(da, d0):.4f}, actual vs row1 {tv(da, d1):.4f}, row0 vs row1 {tv(d0, d1):.4f}')
    lines.append(f'  top slot share of click mass: actual {da[0]:.4f}, row0 {d0[0]:.4f}, row1 {d1[0]:.4f}')
(OUT / f'p_oracle_{name}.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
