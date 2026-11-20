"""How many disjoint 30K holdouts fit in the holdout-eligible set (train.py: p_random > 0.9 and M >= 50)? CPU, seconds.
  CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/recheck/feasibility_value/s5_holdout_pool.py"""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from common import feed_meta
f, M, off = feed_meta('train')
pr = np.load('<repo>/eda/labels_position/out/train_prandom.npz')
p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[M < 50] = 0
cand = np.where((p > 0.9) & (M >= 50))[0]; pool = np.where(p > 0.5)[0]
print('train feeds', len(M), '| holdout-eligible (p_random>0.9, M>=50):', len(cand), '| cooldown pool (p_random>0.5):', len(pool))
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
uid = f['user_id'].to_numpy()
print('holdout users', len(np.unique(uid[hold])), '| holdout feeds whose user has another TRAIN feed outside the holdout:',
      int(np.isin(uid[hold], uid[np.setdiff1d(np.arange(len(M)), hold)]).sum()))
print('disjoint 30K holdouts that fit in the eligible set:', len(cand) // 30000, '| share of the cooldown pool removed by one holdout:', round(30000 / len(pool), 3))
for n in (30000, 60000, 90000, 120000, 150000):  # reference: W8 - N8 on 30K feeds = 0.00028
    print(f'  n={n}: per-feed-noise SE of an 8-vs-8 pool delta {0.00028 * np.sqrt(30000 / n):.5f} (2 SE {2 * 0.00028 * np.sqrt(30000 / n):.5f})')
