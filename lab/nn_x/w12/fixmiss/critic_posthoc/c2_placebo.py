"""Critic check 2 (CPU): (a) placebo for slice offsets: random per-bin offsets with the magnitude the stream fitted
(sd ~0.03-0.05 in z units) on popularity deciles -> how much NDCG does pure noise of that size cost?
(b) overall temperature check: is softmax(s) at T=1 calibrated at item level (E vs O by score decile)?
(c) the short-feed / long-feed contrasts on FD16P20 members, fresh (for the stale-witness claim)."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
import pmix

P = "<repo>/preds"
fv, M, off = feed_meta("val"); n = len(M); fid = np.repeat(np.arange(n), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)


def z(x):
    x = np.asarray(x, np.float64); mu = np.bincount(fid, x, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, n) / M - mu ** 2, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]


s = 0.5 * z(np.load(f"{P}/ens_FD16P20_m04_02kd_ft_val.npy")) + 0.5 * z(np.load(f"{P}/ens_FD16P20_m04_02kd_to_val.npy"))
ev = lambda k: ndcg_per_feed(k, y, off, M)
base = ev(s)
print(f"base {base.mean():.5f}")
F = pl.read_parquet("<repo>/lab/nn_x/w12/fixmiss/posthoc/out/valfeat.parquet", columns=["it_hist_all_luo"])
ih = F["it_hist_all_luo"].to_numpy().astype(np.float64)
qs = np.unique(np.quantile(np.log1p(np.maximum(ih, 0)), np.linspace(0, 1, 11)[1:-1]))
b = np.where(ih <= 0, 0, 1 + np.digitize(np.log1p(np.maximum(ih, 0)), qs))
rng = np.random.default_rng(11)
for sd in (0.02, 0.035, 0.05):
    ds = []
    for r in range(20):
        offv = rng.normal(0, sd, b.max() + 1)
        ds.append((ev(s + offv[b]) - base).mean())
    ds = np.array(ds)
    print(f"placebo popularity-bin offsets sd {sd}: mean d {ds.mean():+.5f}, sd over draws {ds.std():.5f}, min {ds.min():+.5f}, max {ds.max():+.5f}")

# item-level calibration of softmax(s) at T=1 by within-feed score rank bucket and by temperature
mx = np.maximum.reduceat(s, off)
K = np.bincount(fid, y, n)
for T in (0.8, 1.0, 1.25):
    e = np.exp((s - mx[fid]) / T); q = e / np.bincount(fid, e, n)[fid]
    E = K[fid] * q
    r = np.argsort(np.argsort(-s + fid * 1e6))  # global order; recompute within-feed rank
    order = np.lexsort((-s, fid)); rk = np.empty(len(s), np.int64); rk[order] = np.arange(len(s)) - np.repeat(off, M)
    bins = np.digitize(rk, [1, 3, 8, 20, 50, 100])
    O = np.bincount(bins, y); Ee = np.bincount(bins, E)
    ll = (y * np.log(q + 1e-12)).sum() / y.sum()
    print(f"T {T}: O/E by within-feed rank bucket [0,1-2,3-7,8-19,20-49,50-99,100+]:", np.round(O / Ee, 3), f"| mean log q of positives {ll:.4f}")

# FD16P20 members fresh contrasts (short-feed: m0.4_0.2 vs m0.2_0.2; long-feed: m0.2_0.4 vs m0.2_0.2)
S8 = range(951, 959); S4 = range(951, 955)
FD16 = [f'w5_fd10_{a}_s{s_}' for a in ('kd05w192', 'kd05t25w192') for s_ in S8]
FDP = [f'abl_fd10_pop1kd05w192_s{s_}' for s_ in S4]
pool = lambda names, suf: np.mean([z(np.load(f"{P}/nn_{m}_{suf}.npy").astype(np.float32)) for m in names], 0)
ftb, ftp, tob, top = pool(FD16, "n2cpv3s3_val_oof"), pool(FDP, "n2cpv3s3_val_oof"), pool(FD16, "trainonly_val"), pool(FDP, "trainonly_val")
Mr = np.repeat(M, M); big = M >= 50; W = pmix.weights()
def fin(ws, wl):
    w = np.where(Mr >= 50, wl, ws)
    return 0.5 * z((1 - w) * z(ftb) + w * z(ftp)) + 0.5 * z((1 - w) * z(tob) + w * z(top))
e22 = ev(fin(0.2, 0.2)); e42 = ev(fin(0.4, 0.2)); e24 = ev(fin(0.2, 0.4))
f = lambda d, w=None: "{:+.5f}+-{:.5f}".format(*pmix.wmean_se(d, w))
d = e42 - e22; print(f"FD16P20 fresh short contrast m0.4_0.2 - m0.2_0.2: all {f(d)} | pub {f(d, W['public'])} | M<50 {f(d[~big])}   (public +0.00032)")
d = e24 - e22; print(f"FD16P20 fresh long contrast  m0.2_0.4 - m0.2_0.2: all {f(d)} | pub {f(d, W['public'])} | M>=50 {f(d[big])}  (public -0.00015)")
# public share of M>=50 among public test feeds
ft_, Mt, _ = feed_meta("test"); ts = ft_["timestamp"].to_numpy(); pubm = ts < pmix.BOUND
print(f"test feeds {len(Mt)}, public {pubm.sum()} ({pubm.mean():.3f}); share M>=50 in public {np.mean(Mt[pubm] >= 50):.3f}, private {np.mean(Mt[~pubm] >= 50):.3f}, val {big.mean():.3f}")
