"""Review R3: independent re-check of the M-split blend claim (G3 +0.00054 +- 0.00028; pmix +0.00040 +- 0.00040),
user-clustered SE, and a val-vs-test asymmetry the analyses did not consider: the val FT branch is a SINGLE out-of-fold
model per feed, while the test FT branch is the MEAN of the 5 fold models (nn/finetune.py:6,210,272).
Fold-model noise is common to all members (same folds, same FT seed), so the pool cannot average it out on val, but the
5-fold mean removes most of it on test. We measure the fold-noise size on test (per-fold test predictions) by M bin and
estimate, by a noise-injection calibration on val, how much of the val M<50 'FT harm' would not exist on test."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"): os.environ[k] = "12"
import sys; sys.path.insert(0, "<repo>/src")
from common import *
import pmix
P = Path("<repo>/preds")
fv, Mv, offv = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy")
ft, Mt, offt = feed_meta("test")
fidv = np.repeat(np.arange(len(Mv)), Mv); fidt = np.repeat(np.arange(len(Mt)), Mt)
uid = fv["user_id"].to_numpy()

def z(s, fid, M):
    s = s.astype(np.float64); n = len(M)
    mu = np.bincount(fid, s, n) / M; sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]

def nd(s): return ndcg_per_feed(s.astype(np.float32), yv, offv, Mv)

def cl_se(d):  # user-clustered SE (user 0 = singletons)
    g = np.where(uid == 0, -1 - np.arange(len(uid)), uid)
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, d - d.mean()); return np.sqrt((s ** 2).sum()) / len(d)

W = pmix.weights()
G3 = [f"v23_gru_pre2s_s{i}" for i in range(501, 507)] + [f"r6_cd_m02_r{i}" for i in range(101, 108)] + [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)]
for tag in ("G3", "PF7x", "M2"):
    zf = z(np.load(P / f"ens_{tag}_ft_val.npy"), fidv, Mv); zt = z(np.load(P / f"ens_{tag}_to_val.npy"), fidv, Mv)
    fin = 0.5 * zf + 0.5 * zt
    small = (Mv < 50)[fidv]
    ms = np.where(small, zt, fin)
    a, b, c = nd(fin), nd(ms), nd(zt)
    d = b - a
    mp, sp = pmix.wmean_se(d, W["private"])
    print(f"{tag}: final {a.mean():.5f} TO {c.mean():.5f} msplit {b.mean():.5f} | msplit-final {d.mean():+.5f} feedSE {d.std()/np.sqrt(len(d)):.5f} userSE {cl_se(d):.5f} | pmix {mp:+.5f}+-{sp:.5f}"
          f" | M<50 share {np.mean(Mv<50):.3f}; delta on M<50 feeds {d[Mv<50].mean():+.5f}; FT-half on M<20 {(a-c)[Mv<20].mean():+.5f} M20-49 {(a-c)[(Mv>=20)&(Mv<50)].mean():+.5f} M>=50 {(a-c)[Mv>=50].mean():+.5f}")

# ---- fold-noise on test: per-fold pool FT predictions (saved per fold as feed-z, nn/finetune.py:280)
acc = None
for m in G3:
    tf = np.load(P / f"nn_{m}_n2cpv3_test_folds.npy").astype(np.float32)
    acc = tf if acc is None else acc + tf
pool_k = acc / len(G3)                       # (5, n_cand): pool FT branch if built from fold k only
pool_k = np.stack([z(pool_k[k], fidt, Mt) for k in range(5)])
mean_k = z(pool_k.mean(0), fidt, Mt)
zto_t = z(np.load(P / "ens_G3_to_test.npy"), fidt, Mt)
dev2 = ((pool_k - pool_k.mean(0)) ** 2).sum(0) / 4      # unbiased per-candidate variance of a single-fold pool
shift2 = (mean_k - zto_t) ** 2
bins = [(5, 20), (20, 50), (50, 100), (100, 200), (200, 301)]
print("\nTEST, G3 pool: single-fold FT-branch noise sd vs the FT shift (|FT branch - TO branch|), per-candidate RMS in z units")
sig_t = {}
for lo, hi in bins:
    mk = ((Mt >= lo) & (Mt < hi))[fidt]
    sig_t[(lo, hi)] = np.sqrt(dev2[mk].mean())
    print(f"  M {lo:3d}-{hi:3d}: fold-noise sd {np.sqrt(dev2[mk].mean()):.4f}  FT shift rms {np.sqrt(shift2[mk].mean()):.4f}  ratio var {dev2[mk].mean()/shift2[mk].mean():.3f}")
# fold-to-fold correlation of deviations (are fold models' errors independent?)
dv = pool_k - pool_k.mean(0)
print("  mean pairwise corr of fold deviations:", np.round(np.mean([np.corrcoef(dv[i], dv[j])[0, 1] for i in range(5) for j in range(i + 1, 5)]), 3))

# ---- calibration on val: NDCG cost of adding Gaussian noise with that sd to the val FT branch (by M bin), so the
# val OOF FT branch (1 fold model) vs test FT branch (5-fold mean, residual var ~ 1/5 if independent) gap can be sized.
zf = z(np.load(P / "ens_G3_ft_val.npy"), fidv, Mv); zt = z(np.load(P / "ens_G3_to_val.npy"), fidv, Mv)
base = nd(0.5 * zf + 0.5 * zt)
rng = np.random.default_rng(0)
sd_c = np.zeros(len(fidv))
for (lo, hi), s in sig_t.items():
    sd_c[((Mv >= lo) & (Mv < hi))[fidv]] = s
res = []
for r in range(6):
    e = rng.standard_normal(len(fidv)) * sd_c
    zf2 = z(zf + e, fidv, Mv)
    res.append(nd(0.5 * zf2 + 0.5 * zt) - base)
res = np.mean(res, 0)
print("\nVAL calibration: adding one extra unit of single-fold noise to the val FT branch costs (final pool, NDCG):")
for lo, hi in [(5, 50), (50, 301)]:
    mk = (Mv >= lo) & (Mv < hi)
    print(f"  M {lo}-{hi}: {res[mk].mean():+.5f} on those feeds; contribution to overall {res[mk].sum()/len(Mv):+.5f}")
print(f"  overall {res.mean():+.5f}  (if fold noise is independent, test FT carries ~1/5 of the val noise variance, so test gains ~0.8x this back"
      " relative to the val estimate; Gaussian noise is only a proxy for the structured fold noise)")
