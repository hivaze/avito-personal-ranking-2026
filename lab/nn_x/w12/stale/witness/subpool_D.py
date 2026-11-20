"""witness: is D = delta(priv) - delta(lag0) less sensitive to WHICH seeds represent a pool than the deltas themselves?
Test on pair (i) (the only pair with every member scored under both views): random sub-pools 4 of 8 W8 members + 2 of 4 pop1kd members,
statistic S = NDCG(0.8 z(W8 sub) + 0.2 z(pop1kd sub)) - NDCG(z(W8 sub)) under lag0 and priv, D = S_priv - S_lag0.
Used to judge the reduced pools of pair (iv) (DECLARATION block 5: lagged columns reported only).
  cd <repo>/nn_x/w12/stale/witness && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 ../../../../.venv/bin/python subpool_D.py
"""
import os
os.environ.setdefault('WT_THREADS', '4')
from wt_common import *
import lagged_score as LS
L = LS.Lag(); rng = np.random.default_rng(20261115)
W8, P4 = G['W8'], G['pop1kd']
full = L.compare([(0.8, W8), (0.2, P4)], [(1.0, W8)], views=('lag0', 'priv'))
rows = []
for _ in range(60):
    w = [W8[i] for i in sorted(rng.choice(8, 4, replace=False))]; p = [P4[i] for i in sorted(rng.choice(4, 2, replace=False))]
    c = L.compare([(0.8, w), (0.2, p)], [(1.0, w)], views=('lag0', 'priv'))
    rows.append((c['d_lag0'][0], c['d_priv'][0], c['D_priv'][0], c['d_lag0'][1], c['d_priv'][1], c['D_priv'][1]))
r = np.array(rows)
out = [f'=== sub-pool sensitivity of pair (i) on the lagged subset, {now()} (60 random sub-pools: 4 of 8 W8 + 2 of 4 pop1kd) ===',
       f' full pools: S_lag0 {"%+.5f +- %.5f" % full["d_lag0"]} | S_priv {"%+.5f +- %.5f" % full["d_priv"]} | D {"%+.5f +- %.5f" % full["D_priv"]}',
       f' sub-pools : S_lag0 mean {r[:, 0].mean():+.5f} sd {r[:, 0].std(ddof=1):.5f} | S_priv mean {r[:, 1].mean():+.5f} sd {r[:, 1].std(ddof=1):.5f} | D mean {r[:, 2].mean():+.5f} sd {r[:, 2].std(ddof=1):.5f}',
       f' bias of a half-size pool (sub-pool mean - full): S_lag0 {r[:, 0].mean() - full["d_lag0"][0]:+.5f} | S_priv {r[:, 1].mean() - full["d_priv"][0]:+.5f} | D {r[:, 2].mean() - full["D_priv"][0]:+.5f}',
       f' mean per-feed SE of the half-size statistics: S_lag0 {r[:, 3].mean():.5f} | S_priv {r[:, 4].mean():.5f} | D {r[:, 5].mean():.5f}  -> sd over sub-pools / per-feed SE: '
       f'{r[:, 0].std(ddof=1) / r[:, 3].mean():.2f} | {r[:, 1].std(ddof=1) / r[:, 4].mean():.2f} | {r[:, 2].std(ddof=1) / r[:, 5].mean():.2f}',
       f' corr(S_lag0, S_priv) over sub-pools {np.corrcoef(r[:, 0], r[:, 1])[0, 1]:+.3f}']
print('\n'.join(out)); (LOGS / 'subpool_D.txt').write_text('\n'.join(out) + '\n')
