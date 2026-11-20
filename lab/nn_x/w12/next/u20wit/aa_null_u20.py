"""u20wit: same-recipe (A/A) null of the U20 decision statistic, from the 8 finished kd05w192 students (seeds 951-958; stored GPU files).
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit/aa_null_u20.py
Statistic of the rule = member delta paired per feed, pooled over 2 seeds, on the CROSS-USER holdout feeds (24,026) and cross-user val feeds (13,571).
Null draw: 2 "A" seeds and 2 "B" seeds, disjoint, of the SAME recipe (all 420 ordered choices of two disjoint seed pairs; 210 distinct up to sign).
Gives the sd of the statistic under "no effect" (per-feed noise AND seed noise), and from it the chance that the fixed thresholds (+0.0004 / +0.0008) are crossed
for a given true effect. The thresholds are NOT changed; this only says how the verdict should be weighed. Caveat: full-data d192 students stand in for u20a / u20b."""
from wit_common import *
from itertools import combinations
tm = train_meta(); hold = tm['hold']; Mh = tm['M'][hold]; oh = np.concatenate([[0], np.cumsum(Mh)[:-1]]).astype(np.int64); yh = np.load(W5P / 'holdout/holdout_y.npy').astype(np.float32)
crh = tm['xf'][hold] != 0
fv = pl.read_parquet(FLAT / 'val/feeds.parquet'); Mv = fv['M'].to_numpy().astype(np.int64); ov = fv['cand_off'].to_numpy().astype(np.int64); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32)
uv = fv['user_id'].to_numpy().astype(np.int64); tru = np.unique(tm['uid'][tm['uid'] != 0]); v0 = np.isin(uv, tru) & (uv != 0) & ((uv * 2654435761 % 4294967296) % 5 == 0); crv = ~v0
S = list(range(951, 959))
for suf in ('', '@last'):
    Eh = np.array([ndcg_feeds(np.load(W5P / f'holdout/nn_w5_e10_kd05w192_s{s}{suf}_holdout.npy'), yh, oh, Mh)[crh] for s in S])
    Ev = np.array([ndcg_feeds(np.load(W5P / f'nn_w5_e10_kd05w192_s{s}{suf}_trainonly_val.npy'), yv, ov, Mv)[crv] for s in S])
    mh, mv = Eh.mean(1), Ev.mean(1); st_h, st_v, se_h = [], [], []
    for A in combinations(range(8), 2):
        for B in combinations([i for i in range(8) if i not in A], 2):
            st_h.append(mh[list(B)].mean() - mh[list(A)].mean()); st_v.append(mv[list(B)].mean() - mv[list(A)].mean())
            if len(se_h) < 40: se_h.append(se(np.mean([Eh[B[0]] - Eh[A[0]], Eh[B[1]] - Eh[A[1]]], 0)))
    st_h, st_v = np.array(st_h), np.array(st_v)
    print(f'{"selected" if not suf else "@last"} checkpoints, kd05w192 s951-958: member cross-user holdout mean {mh.mean():.5f} (sd across seeds {mh.std(ddof=1):.5f}), cross-user val {mv.mean():.5f} (sd {mv.std(ddof=1):.5f})')
    print(f'  null of the 2-seed pooled member delta: cross-user holdout sd {st_h.std():.5f} (mean paired per-feed SE of one draw {np.mean(se_h):.5f}); cross-user val sd {st_v.std():.5f}; corr(holdout, val) {np.corrcoef(st_h, st_v)[0, 1]:+.2f}; draws {len(st_h)}')
    print(f'  under the null (true B - A = 0): P(statistic < +0.0004) = {np.mean(st_h < 0.0004):.2f} (KILL), P(statistic >= +0.0008) = {np.mean(st_h >= 0.0008):.2f}, P(>= +0.0008 and val >= 0) = {np.mean((st_h >= 0.0008) & (st_v >= 0)):.2f}')
    for d in (0.0004, 0.0008, 0.0012, 0.0016):
        print(f'  true B - A = {d:+.4f} on both witnesses: P(KILL) = {np.mean(st_h + d < 0.0004):.2f}, P(holdout >= +0.0008 and val >= 0) = {np.mean((st_h + d >= 0.0008) & (st_v + d >= 0)):.2f}, P(in between -> 2 more seeds) = {np.mean((st_h + d >= 0.0004) & (st_h + d < 0.0008)):.2f}')
