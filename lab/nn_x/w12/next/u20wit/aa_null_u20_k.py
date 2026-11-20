"""u20wit: same-recipe (A/A) null of the U20 decision statistic for k seeds per arm (k = 2: the read of today; k = 4: the read after "2 more seeds").
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 .venv/bin/python nn_x/w12/next/u20wit/aa_null_u20_k.py
Statistic = member delta paired per feed, pooled over k seeds, on the CROSS-USER holdout feeds (24,026) and cross-user val feeds (13,571); stored GPU files.
Null draw: k "A" seeds and k "B" seeds, disjoint, of the SAME recipe, from the 8 finished students of a wide recipe (kd05w192 and kd05t25w192, seeds 951-958 each;
all ordered choices). Also the 2-POOL / k-POOL delta (mean per-feed z of the k members) under the same draws. The thresholds are NOT changed by this script.
Caveat: full-data d192 students stand in for u20a / u20b (80% of users; the per-feed noise of the fold students is a few % larger)."""
from wit_common import *
from itertools import combinations
from stack_val import feed_rank_z
tm = train_meta(); hold = tm['hold']; Mh = tm['M'][hold]; oh = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64); yh = np.load(W5P / 'holdout/holdout_y.npy').astype(np.float32)
crh = tm['xf'][hold] != 0
fv = pl.read_parquet(FLAT / 'val/feeds.parquet'); Mv = fv['M'].to_numpy().astype(np.int64); ov = fv['cand_off'].to_numpy().astype(np.int64); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32)
uv = fv['user_id'].to_numpy().astype(np.int64); tru = np.unique(tm['uid'][tm['uid'] != 0]); v0 = np.isin(uv, tru) & (uv != 0) & ((uv * 2654435761 % 4294967296) % 5 == 0); crv = ~v0
S = list(range(951, 959))
for rec in ('kd05w192', 'kd05t25w192'):
    ph = [W5P / f'holdout/nn_w5_e10_{rec}_s{s}_holdout.npy' for s in S]; pv = [W5P / f'nn_w5_e10_{rec}_s{s}_trainonly_val.npy' for s in S]
    if not all(p.exists() for p in ph + pv): print(f'{rec}: not all 8 seeds stored, skipped'); continue
    Xh = [np.load(p).astype(np.float32) for p in ph]; Xv = [np.load(p).astype(np.float32) for p in pv]
    mh = np.array([ndcg_feeds(x, yh, oh, Mh)[crh].mean() for x in Xh]); mv = np.array([ndcg_feeds(x, yv, ov, Mv)[crv].mean() for x in Xv])
    Zh = [feed_rank_z(oh, Mh, x)[:, 1] for x in Xh]
    print(f'{rec} s951-958, selected checkpoints: member cross-user holdout {mh.mean():.5f} (sd across seeds {mh.std(ddof=1):.5f}), cross-user val {mv.mean():.5f} (sd {mv.std(ddof=1):.5f})')
    for k in (2, 4):
        st_h, st_v, pl_h = [], [], []; pc = {}
        def pool(ix):
            if ix not in pc: pc[ix] = ndcg_feeds(np.mean([Zh[i] for i in ix], 0), yh, oh, Mh)[crh].mean()
            return pc[ix]
        for A in combinations(range(8), k):
            for B in combinations([i for i in range(8) if i not in A], k):
                st_h.append(mh[list(B)].mean() - mh[list(A)].mean()); st_v.append(mv[list(B)].mean() - mv[list(A)].mean()); pl_h.append(pool(B) - pool(A))
        st_h, st_v, pl_h = np.array(st_h), np.array(st_v), np.array(pl_h)
        print(f'  k = {k} seeds per arm ({len(st_h)} ordered draws): null sd of the pooled member delta: cross-user holdout {st_h.std():.5f}, cross-user val {st_v.std():.5f}; null sd of the {k}-pool delta on the cross-user holdout {pl_h.std():.5f}')
        print(f'    true B - A = 0: P(statistic < +0.0004) = {np.mean(st_h < 0.0004):.2f} (KILL), P(>= +0.0008 and val >= 0) = {np.mean((st_h >= 0.0008) & (st_v >= 0)):.2f}, P(>= +0.00061, the value read today) = {np.mean(st_h >= 0.00061):.3f}')
        for d in (0.0004, 0.0006, 0.0008, 0.0012):
            print(f'    true B - A = {d:+.4f} on both witnesses: P(KILL) = {np.mean(st_h + d < 0.0004):.2f}, P(holdout >= +0.0008 and val >= 0) = {np.mean((st_h + d >= 0.0008) & (st_v + d >= 0)):.2f}, P(in between) = {np.mean((st_h + d >= 0.0004) & (st_h + d < 0.0008)):.2f}')
print('DONE')
