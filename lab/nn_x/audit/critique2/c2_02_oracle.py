"""critique2 N2: reproduce the external audit's 'oracle ceiling' (e02 P6d: val's own in-sample category x day x hour-bucket
positive-rate log-ratio, nested on G3 as z + w*prior, +0.00062 +- 0.00036) with an independent implementation and three
different user-grouped fold seeds; test its sensitivity to the smoothing constant K (30 / 100 / 300) and to the cell
granularity; and quantify how much of the in-sample table is binomial noise. Also the honest OOF cat x day version (P6b).
CPU only. Writes out/c2_02_oracle.log (stdout)."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
import sys, time
from pathlib import Path
import numpy as np, polars as pl
ROOT = Path("<repo>"); sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed
FLAT = ROOT / "data/flat"; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
user = f["user_id"].to_numpy().astype(np.int64); ts = f["timestamp"].to_numpy().astype(np.int64)
n = len(M); fid = np.repeat(np.arange(n), M)
hour = ((ts + 10800) // 3600) % 24; dayv = (ts + 10800) // 86400; dayv = dayv - dayv.min()  # 0 = Tue, 2 = Thu
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
cat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
g3 = np.load(ROOT / "preds/ens_G3_final_val.npy").astype(np.float64)
mu = np.bincount(fid, g3, n) / M; sd = np.sqrt(np.bincount(fid, (g3 - mu[fid]) ** 2, n) / M); sd[sd < 1e-9] = 1
z = (g3 - mu[fid]) / sd[fid]
base = ndcg_per_feed(z, y, off, M); log(f"G3 val {base.mean():.5f}")
groups = np.where(user == 0, -1 - np.arange(n), user)
hb = np.digitize(hour, [7, 12, 18])  # 0-6 / 7-11 / 12-17 / 18-23 as in common_ext.hour_bucket
# train random-like category base rate as in e02 (rate_c): recomputed here from the same definition
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz"); p = np.nan_to_num(pr["p_random"].astype(np.float32))
ft = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M", "cand_off"])
Mt = ft["M"].to_numpy().astype(np.int64); offt = ft["cand_off"].to_numpy().astype(np.int64); p[Mt < 50] = 0
cand_ = np.where((p > 0.9) & (Mt >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand_, min(30000, len(cand_)), replace=False))
is_hold = np.zeros(len(Mt), bool); is_hold[hold] = True
fit = np.where((p > 0.9) & (Mt >= 50) & ~is_hold)[0]
rows = np.concatenate([np.arange(offt[i], offt[i] + Mt[i]) for i in fit])
ctr = np.load(FLAT / "train/cand_category_id.npy", mmap_mode="r")[rows].astype(np.int64)
ytr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[rows].astype(np.float64)
cnt_c = np.bincount(ctr, minlength=64).astype(np.float64); pos_c = np.bincount(ctr, weights=ytr, minlength=64)
rate_c = (pos_c + 10 * ytr.mean()) / (cnt_c + 10)
log(f"train random-like fit rows {len(rows)} from {len(fit)} feeds; base rate {ytr.mean():.5f}")
WG = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]); WG = np.concatenate([-WG[1:][::-1], WG])
def nested(extra, fold):
    nd = np.stack([ndcg_per_feed(z + w * extra, y, off, M) for w in WG])
    oof = np.zeros(n); ws = []
    for k in range(5):
        tr = fold != k; i = int(np.argmax(nd[:, tr].mean(1))); oof[fold == k] = nd[i, fold == k]; ws.append(float(WG[i]))
    d = oof - base
    return d.mean(), d.std(ddof=1) / np.sqrt(n), ws, nd.mean(1).max() - base.mean()
def table_prior(key, ncell, K):
    pos = np.bincount(key, weights=y, minlength=ncell); cnt = np.bincount(key, minlength=ncell).astype(np.float64)
    return pos, cnt
def insample(key, ncell, K, per_cell_rate):
    pos, cnt = table_prior(key, ncell, K)
    sm = (pos + K * per_cell_rate) / (cnt + K)
    return np.log(np.maximum(sm, 1e-6))[key] - np.log(np.maximum(per_cell_rate, 1e-6))[key], pos, cnt
# P6d replication: cat x day(2) x hour bucket(4) = 512 cells, in-sample, K = 100
key = cat * 8 + np.where(dayv[fid] == 0, 0, 4) + hb[fid]
rc8 = np.repeat(rate_c, 8)
for K in [30, 100, 300]:
    x, pos, cnt = insample(key, 512, K, rc8)
    for seed in [0, 1, 2]:
        u, inv = np.unique(groups, return_inverse=True); fold = (np.random.default_rng(seed).permutation(len(u)) % 5)[inv]
        m, se, ws, full = nested(x, fold)
        log(f"P6d in-sample cat x day x hourbucket (512 cells) K={K:3d} foldseed {seed}: OOF {m:+.5f} ± {se:.5f} (z {m/se:+.1f}); w {ws}; best full-fit {full:+.5f}")
# how noisy is the in-sample table? expected binomial variance of the cell log-rate vs its observed between-cell variance
x, pos, cnt = insample(key, 512, 100, rc8)
ok = cnt > 200
obs_lr = np.log(np.maximum((pos[ok] + 100 * rc8[ok]) / (cnt[ok] + 100), 1e-6)) - np.log(rc8[ok])
rate_cell = (pos[ok] + 100 * rc8[ok]) / (cnt[ok] + 100)
binom_var = (1 - rate_cell) / np.maximum(pos[ok] + 100 * rc8[ok], 1) * (cnt[ok] / (cnt[ok] + 100)) ** 2
log(f"in-sample table: {ok.sum()} cells with >200 rows; median positives per cell {np.median(pos[ok]):.0f}; "
    f"observed var of the smoothed log-ratio {obs_lr.var():.4f} vs mean binomial var {binom_var.mean():.4f} "
    f"-> noise share {min(1, binom_var.mean()/max(obs_lr.var(),1e-9)):.2f}")
# honest OOF cat x day (P6b) with 3 fold seeds, and a coarser in-sample cat x day (P6c)
for seed in [0, 1, 2]:
    u, inv = np.unique(groups, return_inverse=True); fold = (np.random.default_rng(seed).permutation(len(u)) % 5)[inv]
    xo = np.zeros(len(y))
    keyd = cat * 2 + np.where(dayv[fid] == 0, 0, 1)
    for k in range(5):
        trm = fold[fid] != k
        pos = np.bincount(keyd[trm], weights=y[trm], minlength=128); cnt = np.bincount(keyd[trm], minlength=128).astype(np.float64)
        sm = (pos + 100 * np.repeat(rate_c, 2)) / (cnt + 100)
        lr = np.log(np.maximum(sm, 1e-6)) - np.log(np.maximum(np.repeat(rate_c, 2), 1e-6))
        xo[~trm] = lr[keyd[~trm]]
    m, se, ws, full = nested(xo, fold)
    log(f"P6b honest user-OOF cat x day (128 cells) foldseed {seed}: OOF {m:+.5f} ± {se:.5f} (z {m/se:+.1f}); w {ws}")
    xi, _, _ = insample(keyd, 128, 100, np.repeat(rate_c, 2))
    m, se, ws, full = nested(xi, fold)
    log(f"P6c in-sample cat x day (128 cells) foldseed {seed}: OOF {m:+.5f} ± {se:.5f} (z {m/se:+.1f}); w {ws}")
# a sharper leaky oracle the audit did not run: cat x day x hour(24) (3072 cells) and cat x hour-bucket x day x M-bucket
key24 = cat * 48 + np.where(dayv[fid] == 0, 0, 24) + hour[fid]
x24, _, _ = insample(key24, 64 * 48, 100, np.repeat(rate_c, 48))
u, inv = np.unique(groups, return_inverse=True); fold = (np.random.default_rng(0).permutation(len(u)) % 5)[inv]
m, se, ws, full = nested(x24, fold)
log(f"extra: in-sample cat x day x hour24 (3072 cells) K=100 foldseed 0: OOF {m:+.5f} ± {se:.5f} (z {m/se:+.1f}); w {ws}; full-fit {full:+.5f}")
log("done")
