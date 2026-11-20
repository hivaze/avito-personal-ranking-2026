"""Pre-training audit A4 (CPU): how many OLDER events a user's earlier snapshot adds beyond the current 256-event window
(history extension by snapshot union). val/test: latest train snapshot behind the val/test snapshot (returning users).
train: for each train feed, the user's earliest earlier train snapshot (users with >=2 feeds; 200K-feed sample)."""
import numpy as np, polars as pl, numba as nb
from pathlib import Path
nb.set_num_threads(12)
R = Path("<repo>"); F = R / "data/flat"; BOUND = 1790497920
def L(s, n): return np.load(F / s / f"{n}.npy", mmap_mode="r")
def keys(s):
    return np.asarray(L(s, "hist_item_id")).astype(np.int64) * 512 + np.asarray(L(s, "hist_event_type")).astype(np.int64) * 32 + np.asarray(L(s, "hist_event_source")).astype(np.int64)
@nb.njit(parallel=True, cache=True)
def ext(kA, oA, HA, kB, oB, HB, out):
    # out[i] = number of events of snapshot B (older) whose key is not in snapshot A (current)
    for i in nb.prange(len(oA)):
        a = kA[oA[i]:oA[i] + HA[i]]; s = set([np.int64(-1)])
        for x in a: s.add(x)
        c = 0
        for x in kB[oB[i]:oB[i] + HB[i]]:
            if x not in s: c += 1
        out[i] = c
fe = {s: pl.read_parquet(F / s / "feeds.parquet") for s in ["train", "val", "test"]}
tr = fe["train"]; ktr = keys("train"); toff = tr["hist_off"].to_numpy().astype(np.int64); tH = tr["H"].to_numpy().astype(np.int64)
lat = tr.filter(pl.col("user_id") != 0).sort("timestamp").group_by("user_id").agg(pl.col("fidx").last())
latd = dict(zip(lat["user_id"].to_list(), lat["fidx"].to_list()))
for s in ["val", "test"]:
    f = fe[s]; k = keys(s); u = f["user_id"].to_numpy(); H = f["H"].to_numpy().astype(np.int64); off = f["hist_off"].to_numpy().astype(np.int64); ts = f["timestamp"].to_numpy()
    idx = np.array([i for i in range(len(u)) if u[i] != 0 and u[i] in latd]); tf = np.array([latd[x] for x in u[idx]])
    out = np.empty(len(idx), np.int64); ext(k, off[idx], H[idx], ktr, toff[tf], tH[tf], out)
    cap = H[idx] == 256
    print(f"{s}: returning feeds {len(idx)} ({len(idx)/len(u):.3f}); older events added by the user's latest train snapshot: mean {out.mean():.1f} q50 {np.median(out):.0f} q90 {np.quantile(out,0.9):.0f}; among capped (H=256) feeds q50 {np.median(out[cap]):.0f}; share of returning feeds with >=32 added {np.mean(out>=32):.3f}; all-feed share with >=32 added {np.sum(out>=32)/len(u):.3f}")
# train parity: each train feed of a multi-feed user vs that user's earliest earlier feed
g = tr.filter(pl.col("user_id") != 0).sort("timestamp").with_columns(pl.col("fidx").first().over("user_id").alias("f0"), pl.len().over("user_id").alias("nf"))
later = g.filter((pl.col("fidx") != pl.col("f0")) & (pl.col("nf") >= 2))
print(f"train feeds that have an earlier same-user train snapshot: {len(later):,} of {len(tr):,} ({len(later)/len(tr):.3f})")
smp = later.sample(n=min(200_000, len(later)), seed=0)
a = smp["fidx"].to_numpy().astype(np.int64); b = smp["f0"].to_numpy().astype(np.int64)
out = np.empty(len(a), np.int64); ext(ktr, toff[a], tH[a], ktr, toff[b], tH[b], out)
print(f"   older events added by the earliest same-user snapshot: mean {out.mean():.1f} q50 {np.median(out):.0f} q90 {np.quantile(out,0.9):.0f}; share >=32 {np.mean(out>=32):.3f}; all-train-feed share with >=32 added {np.sum(out>=32)/len(smp)*len(later)/len(tr):.3f}")
