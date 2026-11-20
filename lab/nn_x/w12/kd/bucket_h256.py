"""Secondary read declared 2026-11-15T17:03Z for kwr32 (information for any arm): N-pool delta vs the reference arm on feeds at the history cap (H = 256) and on the rest.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/kd/bucket_h256.py <arm> <seeds csv> [ref arm = kd05w192]
Train-only pools (member-equal per-feed rank-z), selected checkpoints, paired per feed, val and the in-period holdout. CPU, read-only.
Holdout feed rows = the fixed 30,000 of train.py --holdout 30000 (same draw as nn_x/w12/seg/refill/align_lib.py)."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
import polars as pl
arm = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(',')]; ref = sys.argv[3] if len(sys.argv) > 3 else 'kd05w192'
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
Hv = m.fv['H'].to_numpy().astype(np.int64)
ft = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['H', 'M']); Mtr = ft['M'].to_numpy().astype(np.int64)
pr = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0); pr[Mtr < 50] = 0.0
hidx = np.sort(np.random.default_rng(12345).choice(np.where((pr > 0.9) & (Mtr >= 50))[0], 30000, replace=False))
Hh = ft['H'].to_numpy().astype(np.int64)[hidx]; assert (Mtr[hidx] == m.Mh).all(), 'holdout rows do not match holdout_meta'
def pool(a):
    V, Hd = [], []
    for s in seeds:
        pv, ph, _ = paths(f'w5_e10_{a}_s{s}'); V.append(m.z('val', np.load(pv).astype(np.float32))); Hd.append(m.z('hold', np.load(ph).astype(np.float32)))
    return ev(np.mean(V, 0)), eh(np.mean(Hd, 0))
av, ah = pool(arm); rv, rh = pool(ref); dv, dh = av - rv, ah - rh
f = lambda d: f'{d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f} (z {d.mean() / (d.std() / np.sqrt(len(d))):+.2f}, n {len(d)})'
print(f'{arm} - {ref}, {len(seeds)}-pool, seeds {seeds}')
for nm, mv, mh in (('H = 256', Hv == 256, Hh == 256), ('H < 256', Hv < 256, Hh < 256), ('all', Hv >= 0, Hh >= 0)):
    print(f'  {nm:8s}: val {f(dv[mv])} | holdout {f(dh[mh])}')
