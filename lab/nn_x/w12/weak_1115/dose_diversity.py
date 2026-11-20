"""Read-only CPU description (auditor, 2026-11-15): per arm of stored 10-epoch students (train-only, selected checkpoints):
member mean NDCG, N-pool NDCG, mean pairwise correlation of the members' per-feed z vectors (val and in-period holdout).
Purpose: is the weight-decay dose-response (0.02 / 0.05 / 0.10, monotone harm at pool level on record) a DIVERSITY effect
(members up, pool down), so that the un-run dose 0 is the open end of a monotone curve?  One look, no selection: arms are named below.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/weak_1115/dose_diversity.py
Matched seeds 951-954 for every arm (4-pool), so the pools are comparable; the 8-seed arms are also printed at 8."""
import os, sys, itertools
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
ARMS = [('kd05w192', 'wd 0.02 / drop 0.1 / kd_w 0.5 (reference)'), ('kwwd05', 'wd 0.05'), ('kwwd10', 'wd 0.10'),
        ('kwd2', 'dropout 0.2'), ('kd08w192', 'kd_w 0.8'), ('kwaux0', 'aux 0'), ('kwh6', '6 heads'), ('kwr32', 'retrieval 32'), ('kd05w224', 'width 224')]
def load(v):
    pv, ph, _ = paths(v); return m.z('val', np.load(pv).astype(np.float32)), m.z('hold', np.load(ph).astype(np.float32))
def desc(arm, seeds):
    Z = [load(f'w5_e10_{arm}_s{s}') for s in seeds]
    out = []
    for i, e in ((0, ev), (1, eh)):
        zs = [z[i] for z in Z]; mem = np.mean([e(z).mean() for z in zs]); pool = e(np.mean(zs, 0)).mean()
        cc = np.mean([np.corrcoef(a, b)[0, 1] for a, b in itertools.combinations(zs, 2)])
        out.append((mem, pool, cc))
    return out
print('arm | seeds | val: member / pool / pair corr | holdout: member / pool / pair corr')
for arm, lab in ARMS:
    for seeds in ((951, 952, 953, 954), tuple(range(951, 959))):
        try: (vm, vp, vc), (hm, hp, hc) = desc(arm, seeds)
        except FileNotFoundError: continue
        print(f'{arm:10s} {lab:42s} | n {len(seeds)} | {vm:.5f} / {vp:.5f} / {vc:.4f} | {hm:.5f} / {hp:.5f} / {hc:.4f}', flush=True)
