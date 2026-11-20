# Calibration of the rule above. The 16 students of one recipe are split at random into two pools of 8 (and of
# 4), 400 times, and the difference between the two pools is read exactly like an experiment.
# Run on CPU with 12 threads; it reads stored predictions and prints to stdout (no arguments).
# Output: ../results/pooled_z_rule_null.txt. Under this null the pooled z has sd 0.80, and
# 'both differences >= 0 and pooled z >= 2' passes 0.25% of the splits.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/recheck/synthesis/aa_null.py
"""SYNTHESIS re-check 11-14: A/A null of the adopted pooled-z rule, from the 16 finished wide students (one recipe).
Random disjoint N|N splits of the 16 (N = 8 and 4): pool A minus pool B, train-only level, paired per feed on val and holdout.
Question: is the per-feed paired SE the right yardstick when seeds also vary? Reports the sd of z_val, z_hold and pooled z under
the null and the empirical pass rate of 'both deltas >= 0 AND (z_val + z_hold)/sqrt(2) >= thr'.
CPU only (12 threads); reads predictions; writes stdout only."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
import numba as nb, time
W = Path('<repo>/nn_x/w5/preds')
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32)
Mv = np.asarray(Mv, np.int64); ov = np.asarray(ov, np.int64)
Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(W / 'holdout/holdout_y.npy').astype(np.float32)
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]; zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]

@nb.njit(parallel=True, cache=False)
def nd(s, y, off, M):  # per-feed full-list NDCG, binary gain, pessimistic ties (negatives first among equal scores)
    out = np.ones(len(M))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        t = y[o:o + m]; v = s[o:o + m]
        o0 = np.argsort(t, kind='mergesort')            # negatives first
        o1 = np.argsort(-v[o0], kind='mergesort')       # stable: score desc, ties keep negatives first
        ts = t[o0][o1]
        npos = 0; dcg = 0.0
        for k in range(m):
            if ts[k] > 0:
                dcg += 1.0 / np.log2(k + 2.0); npos += 1
        if npos > 0:
            idcg = 0.0
            for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
            out[f] = dcg / idcg
    return out

names = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 959)]
ZV = np.stack([zv(np.load(W / f'nn_{v}_trainonly_val.npy')) for v in names]).astype(np.float32)
ZH = np.stack([zh(np.load(W / f'holdout/nn_{v}_holdout.npy')) for v in names]).astype(np.float32)
ev = lambda s: nd(np.ascontiguousarray(s, np.float32), yv, ov, Mv); eh = lambda s: nd(np.ascontiguousarray(s, np.float32), yh, oh, Mh)
p16 = ZV.mean(0); chk = ndcg_per_feed(p16, yv, ov, Mv).mean()
print('members', len(names), 'val rows', ZV.shape[1], 'holdout rows', ZH.shape[1])
print(f'W16 TO pool: val {ev(p16).mean():.5f} (src/metrics.py {chk:.5f}) holdout {eh(ZH.mean(0)).mean():.5f}', flush=True)
def stat(ia, ib):
    dv = ev(ZV[ia].mean(0)) - ev(ZV[ib].mean(0)); dh = eh(ZH[ia].mean(0)) - eh(ZH[ib].mean(0))
    return dv.mean(), dv.std() / np.sqrt(len(dv)), dh.mean(), dh.std() / np.sqrt(len(dh))
rng = np.random.default_rng(1114)
for N, nrep in ((8, 400), (4, 400)):
    R = []; t0 = time.time()
    for r in range(nrep):
        p = rng.permutation(16); R.append(stat(p[:N], p[N:2 * N]))
    R = np.array(R); zv_, zh_ = R[:, 0] / R[:, 1], R[:, 2] / R[:, 3]
    zv2 = np.concatenate([zv_, -zv_]); zh2 = np.concatenate([zh_, -zh_]); pz = (zv2 + zh2) / np.sqrt(2)  # A-B and B-A
    print(f'\n== {N}|{N} random disjoint splits of the 16 wide students, {nrep} splits ({time.time() - t0:.0f} s)')
    print(f' per-feed paired SE (mean): val {R[:, 1].mean():.5f} holdout {R[:, 3].mean():.5f}')
    print(f' sd of the pool delta over splits: val {R[:, 0].std():.5f} holdout {R[:, 2].std():.5f}  (ratio to the per-feed SE: {R[:, 0].std() / R[:, 1].mean():.2f} / {R[:, 2].std() / R[:, 3].mean():.2f})')
    print(f' sd of z_val {zv_.std():.2f}, z_hold {zh_.std():.2f}, pooled z {pz.std():.2f}; corr(z_val, z_hold) over splits {np.corrcoef(zv_, zh_)[0, 1]:+.2f}')
    print(f' P(both deltas >= 0) {np.mean((zv2 >= 0) & (zh2 >= 0)):.3f}; quantiles of pooled z: 95% {np.quantile(pz, 0.95):+.2f}, 97.5% {np.quantile(pz, 0.975):+.2f}, 99% {np.quantile(pz, 0.99):+.2f}; share of pooled z >= 1.60 (the W8 - NEW8 reading): {np.mean(pz >= 1.60):.4f}')
    for thr in (1.3, 1.6, 2.0, 2.5, 3.0):
        print(f' pass rate [both >= 0 and pooled z >= {thr}]: {np.mean((zv2 >= 0) & (zh2 >= 0) & (pz >= thr)):.4f}   (pooled z >= {thr} alone: {np.mean(pz >= thr):.4f})', flush=True)
a = stat(np.arange(8, 16), np.arange(8))
print(f'\n== kd05t25w192 x8 minus kd05w192 x8 (same seeds 951-958): val {a[0]:+.5f} +- {a[1]:.5f} (z {a[0] / a[1]:+.2f}) holdout {a[2]:+.5f} +- {a[3]:.5f} (z {a[2] / a[3]:+.2f}) pooled z {(a[0] / a[1] + a[2] / a[3]) / np.sqrt(2):+.2f}')
a = stat(np.r_[0:4, 8:12], np.r_[4:8, 12:16])
print(f'== W8 (seeds 951-954) minus NEW8 (seeds 955-958): val {a[0]:+.5f} +- {a[1]:.5f} (z {a[0] / a[1]:+.2f}) holdout {a[2]:+.5f} +- {a[3]:.5f} (z {a[2] / a[3]:+.2f}) pooled z {(a[0] / a[1] + a[2] / a[3]) / np.sqrt(2):+.2f}')
for k in range(4):  # kd05w192 4-pool of seeds 951-954 (the 18:24Z reference) vs the matched-N mean
    pass
i4 = np.arange(4)
print(f'== kd05w192 s951-954 4-pool: val {ev(ZV[i4].mean(0)).mean():.5f} holdout {eh(ZH[i4].mean(0)).mean():.5f}; kd05w192 s951-958 8-pool: val {ev(ZV[:8].mean(0)).mean():.5f} holdout {eh(ZH[:8].mean(0)).mean():.5f}')
