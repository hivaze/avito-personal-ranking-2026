"""Critic re-check of two headline numbers, independent of quantify/: (1) val NDCG of the best file (0.5 ft + 0.5 to),
(2) exact expected NDCG on K=1 feeds if the click were drawn from softmax(s/T) (T = 1.0036, quantify's half-A fit),
and the same for all feeds treating each feed's first click only (K forced to 1 on the actual feeds).
Usage: python c_recheck.py -> out/c_recheck.txt"""
import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, '<repo>')
from avito_ranker.metrics import ndcg_per_feed

R = Path('<repo>'); OUT = R / 'lab/nn_x/w12/noise/critic/out'; OUT.mkdir(exist_ok=True)
f = pl.read_parquet(R / 'artifacts/data/flat/val/feeds.parquet'); M = f['M'].to_numpy().astype(np.int64)
off = np.cumsum(M) - M
y = np.load(R / 'artifacts/data/flat/val/cand_target.npy').astype(np.float64)
s = 0.5 * np.load(R / 'artifacts/predictions/ens_FD16P20_m04_02kd_ft_val.npy').astype(np.float64) + \
    0.5 * np.load(R / 'artifacts/predictions/ens_FD16P20_m04_02kd_to_val.npy').astype(np.float64)
nd = ndcg_per_feed(s, y, off, M)
fe = np.repeat(np.arange(len(M)), M); K = np.add.reduceat(y, off)
T = 1.0036
z = s / T; mx = np.maximum.reduceat(z, off); e = np.exp(z - mx[fe]); p = e / np.add.reduceat(e, off)[fe]
order = np.lexsort((-s, fe)); rk = np.empty(len(s), np.int64); rk[order] = np.arange(len(s)) - off[fe[order]]
disc = 1.0 / np.log2(rk + 2.0)
expK1 = np.add.reduceat(p * disc, off)  # exact E[NDCG] of a single click drawn from p
k1 = K == 1
L = [f'val feeds {len(M)}, rows {len(s)}, positives {int(y.sum())}; best file NDCG {nd.mean():.5f} ± {nd.std(ddof=1)/np.sqrt(len(nd)):.5f}',
     f'K=1 feeds {k1.sum()} ({k1.mean():.3f}): actual {nd[k1].mean():.5f}, exact expected under softmax(s/{T}) {expK1[k1].mean():.5f}, diff {nd[k1].mean()-expK1[k1].mean():+.5f} ± {(nd[k1]-expK1[k1]).std(ddof=1)/np.sqrt(k1.sum()):.5f}',
     f'per-feed draw variance (K=1 feeds, exact): mean {np.add.reduceat(p*disc**2, off)[k1].mean() - (expK1[k1]**2).mean():.5f}; observed per-feed var {nd[k1].var():.5f}']
(OUT / 'c_recheck.txt').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
