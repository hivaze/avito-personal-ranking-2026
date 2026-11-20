"""Hidden display position in test-like feeds, measured out of sample: a student that never saw the 30K random-order holdout
scores it with the random-order domain row (as on val/test); an oracle adds log e(display slot) estimated on OTHER
random-order train feeds (within-feed estimator).  Also an oracle that knows the last displayed item (logs end at a click,
so this one partly reveals the click; reported for scale, not as noise).
Usage: CUDA_VISIBLE_DEVICES="" python p_holdout_oracle.py w5_e10_kd05w192_s951  -> out/p_holdout_oracle_<ckpt>.txt
"""
import sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, '<repo>')
from avito_ranker.model import load_ranker
from avito_ranker.data.loader import load_split
from avito_ranker.training.predict import predict
from avito_ranker.training.train import read_exploration_probability, draw_holdout
from avito_ranker.metrics import ndcg_per_feed

torch.set_num_threads(12)
ROOT = Path('<repo>'); OUT = ROOT / 'lab/nn_x/w12/noise/synthesis/out'
name = sys.argv[1]; t0 = time.time()
model = load_ranker(ROOT / f'artifacts/members/checkpoints/{name}.pt', 'cpu')
split = load_split('train', model.config.view, 'cpu')
M = split.feed_sizes; off = split.candidate_offsets; tgt = split.targets
pos = np.load(ROOT / 'artifacts/data/flat/train/cand_pos.npy', mmap_mode='r')
p_v2 = np.load(ROOT / 'lab/eda/labels_position/out/train_prandom_v2.npz')['p_random'].astype(np.float32)
p_old = np.nan_to_num(np.load(ROOT / 'artifacts/exploration/train_prandom.npz')['p_random'].astype(np.float32), nan=-1)
hold = draw_holdout(read_exploration_probability(M), M, 30000)
clean = hold[p_v2[hold] > 0.9]
ref = np.setdiff1d(np.flatnonzero((M >= 50) & (p_old > 0.9) & (p_v2 > 0.9)), hold)
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 199), (200, 10**6)]; NB = len(BINS)


def layout(fids):
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fids])
    fidx = np.repeat(np.arange(len(fids)), M[fids])
    order = np.lexsort((np.asarray(pos[rows]), fidx))
    start = np.concatenate([[0], np.cumsum(M[fids])[:-1]])
    rk = np.empty(len(rows), np.int64); rk[order] = np.arange(len(rows)) - np.repeat(start, M[fids])
    b = np.full(len(rows), -1)
    for k, (lo, hi) in enumerate(BINS):
        b[(rk >= lo) & (rk <= hi)] = k
    return rows, b, rk == np.repeat(M[fids], M[fids]) - 1, start


rr, rb, rl, rst = layout(ref); ry = tgt[rr].astype(np.float64)
feed_rate = np.add.reduceat(ry * ~rl, rst) / (M[ref] - 1); exp_r = np.repeat(feed_rate, M[ref])
e = np.array([ry[(rb == k) & ~rl].sum() / exp_r[(rb == k) & ~rl].sum() for k in range(NB)])
lastr, nonlast = ry[rl].mean(), ry[~rl].mean()
s = predict(model, split, clean)  # no domain mix -> inference domain (random-order row), as for val/test
rows, b, last, start = layout(clean); y = tgt[rows].astype(np.float64); sc = s[rows].astype(np.float64)
base = ndcg_per_feed(sc, y, start, M[clean])
lines = [f'checkpoint {name} (holdout excluded from its training); inference domain {model.inference_domain}',
         f'holdout {len(hold):,} feeds, clean (v2 p>0.9) {len(clean):,}; e from {len(ref):,} other random-order feeds (M>=50)',
         f'within-feed relative examination by slot bin {[f"{lo}-{hi}" if hi < 10**6 else f"{lo}+" for lo, hi in BINS]}: {np.round(e, 3).tolist()}',
         f'last displayed item click rate {lastr:.4f} vs non-last {nonlast:.4f}; share of clicks on last item in holdout {y[last].sum()/y.sum():.3f}',
         f'member NDCG on clean holdout: {base.mean():.5f}']
for tag, off_ in (('slot examination only', np.log(e)[b]), ('last displayed item only', np.where(last, np.log(lastr / nonlast), 0.0)),
                  ('both', np.log(e)[b] + np.where(last, np.log(lastr / nonlast), 0.0))):
    for scale in ((0.5, 1.0) if tag == 'slot examination only' else (1.0,)):
        d = ndcg_per_feed(sc + scale * off_, y, start, M[clean]) - base
        lines.append(f'  oracle {tag} (offset x{scale}): {d.mean():+.5f} +- {d.std(ddof=1)/np.sqrt(len(d)):.5f}')
lo_ = ndcg_per_feed(sc + np.where(last, np.log(lastr / nonlast), 0.0), y, start, M[clean])
bo_ = ndcg_per_feed(sc + np.log(e)[b] + np.where(last, np.log(lastr / nonlast), 0.0), y, start, M[clean])
d = bo_ - lo_
lines.append(f'  slot examination on top of the last-item oracle (both - last only): {d.mean():+.5f} +- {d.std(ddof=1)/np.sqrt(len(d)):.5f}')
K = np.add.reduceat(y, start); k1 = K[np.repeat(np.arange(len(clean)), M[clean])] == 1
lines.append(f'  K=1 feeds whose click is the last displayed item: {(y * last * k1).sum() / (K == 1).sum():.3f} of {(K == 1).sum():,} K=1 feeds')
lines.append(f'wall {time.time()-t0:.0f}s')
(OUT / f'p_holdout_oracle_{name}.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
