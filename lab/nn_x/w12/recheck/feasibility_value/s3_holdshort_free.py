"""FREE form of HOLD-SHORT (lb_stats#5, pool_composition#12): no model inference. The stored full-feed holdout predictions are
restricted to a random sub-feed (size m drawn from the val M<50 distribution; one random positive forced in, the other m-1 rows
uniform), z-scored inside the sub-feed, pooled, scored. kd_audit#10 / kd_literature#2 measured that re-scoring a sub-feed gives
logits at within-feed corr 0.9978 (KL 0.002-0.005 nats, 10x below seed noise) to the restricted full-feed logits, so this is
what a re-scored HOLD-SHORT would show up to that second-order term, and it carries no stale-context artefact.
Question for the feasibility review: does the instrument say anything the full holdout does not?
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/recheck/feasibility_value/s3_holdshort_free.py"""
import numpy as np
from fv_common import *

S = (951, 952, 953, 954); R = 4  # replicates of the sub-sampling
arms = ('base', 'kd05', 'kd05t25', 'kd05w192', 'kd05t25w192')
raw_h = {a: [np.load(W / f'holdout/nn_w5_e10_{a}_s{s}_holdout.npy').astype(np.float32) for s in S] for a in arms}
raw_v = {a: [np.load(W / f'nn_w5_e10_{a}_s{s}_trainonly_val.npy').astype(np.float32) for s in S] for a in arms}
pools = {'base4': ['base'], 'kd05x4': ['kd05'], 'N8': ['kd05', 'kd05t25'], 'W8': ['kd05w192', 'kd05t25w192'],
         'S16': ['kd05', 'kd05t25', 'kd05w192', 'kd05t25w192']}
contrasts = [('kd05x4', 'base4', 'KD (d128, 4 vs 4)'), ('W8', 'N8', 'width under KD (8 vs 8)'), ('W8', 'base4', 'W8 vs hard-label 4-pool'),
             ('S16', 'W8', 'S16 vs W8')]


def pool_nd(raw, off, M, y, sel=None):
    out = {}
    for p, al in pools.items():
        zs = [feed_rank_z(off, M, (x if sel is None else x[sel]))[:, 1] for a in al for x in raw[a]]
        out[p] = ndcg_per_feed(np.mean(zs, 0), y, off, M)
    return out


full = pool_nd(raw_h, oh, Mh, yh)
val = pool_nd(raw_v, ov, Mv, yv)
mshort = Mv[Mv < 50]
rng = np.random.default_rng(20261114)
fid = np.repeat(np.arange(len(Mh)), Mh)
short = {p: np.zeros(len(Mh)) for p in pools}; msum = np.zeros(len(Mh))
for r in range(R):
    m = rng.choice(mshort, len(Mh))  # target size per holdout feed (all holdout feeds have M >= 50 > m)
    key = rng.random(len(yh))
    # force one random positive per feed: the positive with the smallest key gets key -1
    pk = np.where(yh > 0, key, 2.0); o = np.lexsort((pk, fid)); first = o[oh]  # row of the smallest positive key per feed
    assert (yh[first] > 0).all()
    key[first] = -1.0
    o = np.lexsort((key, fid)); rank = np.empty(len(key), np.int64); rank[o] = np.arange(len(key)) - np.repeat(oh, Mh)
    keep = rank < np.repeat(m, Mh); sel = np.flatnonzero(keep)
    Ms = m.astype(np.int64); os_ = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64); ys = yh[sel]
    nd = pool_nd(raw_h, os_, Ms, ys, sel)
    for p in pools:
        short[p] += nd[p] / R
    msum += m / R
    print(f'replicate {r}: sub-feed mean M {m.mean():.1f}, positives per sub-feed {np.add.reduceat(ys, os_).mean():.2f} '
          f'(val M<50: mean M {mshort.mean():.1f}, positives {np.add.reduceat(yv, ov)[Mv < 50].mean():.2f}); W8 NDCG {nd["W8"].mean():.5f}', flush=True)
print(f'\nlevels: W8 full holdout {full["W8"].mean():.5f} | W8 HOLD-SHORT(free) {short["W8"].mean():.5f} | W8 val M<50 {val["W8"][Mv < 50].mean():.5f} | val M>=50 {val["W8"][Mv >= 50].mean():.5f}')
print(f'\n{"contrast":28s} | {"holdout full (30K, M>=50)":26s} | {"HOLD-SHORT free (30K x {} reps)".format(R):30s} | {"val M<50 (5.9K)":22s} | {"val M>=50 (8.5K)":22s} | corr(full, short) per feed')
sm = Mv < 50
for a, b, nm in contrasts:
    df = full[a] - full[b]; ds = short[a] - short[b]; dv = val[a] - val[b]
    print(f'{nm:28s} | {fmt(df):26s} | {fmt(ds):30s} | {fmt(dv[sm]):22s} | {fmt(dv[~sm]):22s} | {np.corrcoef(df, ds)[0, 1]:.3f}')
print('\nReading: if HOLD-SHORT(free) deltas equal the full-holdout deltas times a constant and the per-feed correlation is high, the instrument')
print('re-weights the same information; it cannot confirm or refute a val M<50 effect that the full holdout does not already show.')
