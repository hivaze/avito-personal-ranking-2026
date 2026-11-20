"""Critique check 4: the features-audit numbers behind plan items 6 / 7 / L3.
 (a) F7x 4-member TO pool (r6_cd_m02_r101-104): stored GPU stale-val ('both_pre2both' = the PESSIMISTIC emulation of
     review_shift: train ends Sep 14 AND no future snapshots) vs the stored train-only val preds -> pool delta, pmix,
     by M, by mean listing age of the feed's positives (7-24 d is the audit's target bucket).
 (b) G3 (blend / TO / FT) act/exp by listing-age bin and the cross-fitted 8-bin post-hoc offset (avito a07 bound).
CPU only."""
import os, sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ.setdefault(k, "12")
sys.path.insert(0, "<repo>/src")
import numpy as np
from common import feed_meta, FLAT, FEATS
from metrics import ndcg_per_feed
import pmix
ROOT = "<repo>"
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy").astype(np.float64); n = len(M)
fid = np.repeat(np.arange(n), M)
W = pmix.weights()
knots = np.load(FEATS / "life_f4_age_knots.npy"); K = 64; CAP = 63 / 64 - 1e-6
def age_hours(x):
    x = x.astype(np.float64); u = x * K; k = np.clip(np.floor(u).astype(int), 1, K - 2); w = u - k
    return np.where(x < CAP, knots[k] * (1 - w) + knots[k + 1] * w, np.nan)
age = age_hours(np.load(FLAT / "val/cand_f4.npy"))
def z(s):
    mu = np.bincount(fid, s, n) / M; sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 1e-12)); return (s - mu[fid]) / sd[fid]
def se(d, w=None):
    return pmix.wmean_se(d, w)
# mean age (days) of each feed's positives
ad = age / 24.0; pos_age = np.where(y > 0, np.nan_to_num(ad, nan=np.nanmedian(ad)), 0.0)
mean_pos_age = np.bincount(fid, pos_age, n) / np.maximum(np.bincount(fid, y, n), 1)
EDGES = [0, 1, 3, 7, 24, 76, 1e9]; LAB = ["<1d", "1-3d", "3-7d", "7-24d", "24-76d", ">76d"]
ab = np.digitize(mean_pos_age, EDGES[1:-1])

# (a) stale replay of the F7x TO pool
roots = [101, 102, 103, 104]
live = [np.load(f"{ROOT}/preds/nn_r6_cd_m02_r{r}_trainonly_val.npy").astype(np.float64) for r in roots]
stale = [np.load(f"{ROOT}/nn_x/r6/preds/stale/nn_r6_cd_m02_r{r}_staleval.npy").astype(np.float64) for r in roots]
pl_ = ndcg_per_feed(np.mean([z(s) for s in live], 0).astype(np.float32), y, off, M)
ps_ = ndcg_per_feed(np.mean([z(s) for s in stale], 0).astype(np.float32), y, off, M)
d = ps_ - pl_
print(f"F7x TO 4-pool: live {pl_.mean():.5f}, stale(both_pre2both) {ps_.mean():.5f}, delta {d.mean():+.5f} ± {se(d)[1]:.5f}; pmix-private {se(d, W['private'])[0]:+.5f} ± {se(d, W['private'])[1]:.5f}; pmix-public {se(d, W['public'])[0]:+.5f} ± {se(d, W['public'])[1]:.5f}")
for r, a, b in zip(roots, live, stale):
    dd = ndcg_per_feed(b.astype(np.float32), y, off, M) - ndcg_per_feed(a.astype(np.float32), y, off, M)
    print(f"  r{r}: member stale delta {dd.mean():+.5f} ± {se(dd)[1]:.5f}")
sm = M < 50
print(f"  by M: M<50 {d[sm].mean():+.5f} ± {d[sm].std()/np.sqrt(sm.sum()):.5f}; M>=50 {d[~sm].mean():+.5f} ± {d[~sm].std()/np.sqrt((~sm).sum()):.5f}")
tot = 0.0
for i, lab in enumerate(LAB):
    m = ab == i; c = d[m].mean() * m.mean(); tot += c
    print(f"  positives' mean age {lab:7s}: feeds {m.sum():5d} ({m.mean():.3f}) delta {d[m].mean():+.5f} ± {d[m].std()/np.sqrt(m.sum()):.5f}  contribution to pool delta {c:+.5f}; private weight share {W['private'][m].sum()/W['private'].sum():.3f}")
print(f"  (sum of contributions {tot:+.5f})")

# (b) act/exp by age bin on G3 and the cross-fitted 8-bin offset
to = np.load(f"{ROOT}/preds/ens_G3_to_val.npy").astype(np.float64); ft = np.load(f"{ROOT}/preds/ens_G3_ft_val.npy").astype(np.float64)
blend = 0.5 * z(to) + 0.5 * z(ft)
bins = [0, 3, 12, 24, 72, 336, 2160, 8760, 1e9]; labs = ["<3h", "3-12h", "12-24h", "1-3d", "3-14d", "2w-3m", "3m-1y", ">1y"]
ab8 = np.digitize(np.nan_to_num(age, nan=1e8), bins[1:-1])
def softmax_feed(s):
    mx = np.zeros(n) - 1e9; np.maximum.at(mx, fid, s); e = np.exp(s - mx[fid]); den = np.bincount(fid, e, n); return e / den[fid]
def actexp(s):
    p = softmax_feed(s) * np.bincount(fid, y, n)[fid]  # expected positives under a per-feed softmax calibrated to npos
    return [y[ab8 == b].sum() / max(p[ab8 == b].sum(), 1e-9) for b in range(8)]
print("act/exp by age bin " + str(labs))
for nm, s in (("blend", blend), ("TO", z(to)), ("FT", z(ft))):
    print(f"  {nm:5s}: " + " ".join(f"{v:.3f}" for v in actexp(s)) + f" | row share " + " ".join(f"{np.mean(ab8 == b):.3f}" for b in range(8)))
# cross-fitted offsets (user-grouped 5 folds): offset_b = log(act/exp) fitted on 4 folds, applied to the 5th
uid = f["user_id"].to_numpy(); rng = np.random.default_rng(0); users = np.unique(uid); fold_of_user = dict(zip(users, rng.integers(0, 5, len(users))))
fold = np.array([fold_of_user[u] for u in uid]); base_nd = ndcg_per_feed(blend.astype(np.float32), y, off, M)
for w in (0.5, 1.0):
    new = blend.copy()
    for k in range(5):
        tr_rows = fold[fid] != k
        p = softmax_feed(blend) * np.bincount(fid, y, n)[fid]
        offs = np.array([np.log(max(y[tr_rows & (ab8 == b)].sum(), 1) / max(p[tr_rows & (ab8 == b)].sum(), 1e-9)) for b in range(8)])
        te_rows = fold[fid] == k; new[te_rows] = blend[te_rows] + w * offs[ab8[te_rows]]
    dd = ndcg_per_feed(new.astype(np.float32), y, off, M) - base_nd
    print(f"  post-hoc 8-bin age offset w={w}: OOF {dd.mean():+.5f} ± {se(dd)[1]:.5f}; pmix-private {se(dd, W['private'])[0]:+.5f} ± {se(dd, W['private'])[1]:.5f}; M<50 {dd[sm].mean():+.5f}, M>=50 {dd[~sm].mean():+.5f}")
print("done c04")
