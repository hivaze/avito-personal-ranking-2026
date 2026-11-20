"""Label-free counts: feeds per split with >= 1 strictly earlier snapshot of the same user (user_id != 0), capped or not,
gap to the most recent earlier snapshot, and how many have one >= k days back. No label is read."""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
import numpy as np
import polars as pl
FLAT = "<repo>/data/flat"
SPL = ["train", "val", "test"]
D = {}
for i, s in enumerate(SPL):
    f = pl.read_parquet(f"{FLAT}/{s}/feeds.parquet")
    D[s] = (f["user_id"].to_numpy().astype(np.int64), f["timestamp"].to_numpy().astype(np.int64), f["H"].to_numpy())
u = np.concatenate([D[s][0] for s in SPL]); t = np.concatenate([D[s][1] for s in SPL])
sp = np.concatenate([np.full(len(D[s][0]), i) for i, s in enumerate(SPL)])
o = np.lexsort((t, u)); us, ts, ss = u[o], t[o], sp[o]
for i, s in enumerate(SPL):
    uu, tt, H = D[s]
    n = len(uu)
    lo = np.searchsorted(us, uu)  # first row of the user
    # number of rows of the same user with ts < own ts  (strictly earlier) -> searchsorted on (u, t)
    key = us * (1 << 32) + ts
    k_own = uu * (1 << 32) + tt
    pos = np.searchsorted(key, k_own, side="left")
    n_earlier = pos - lo
    ok = (uu != 0) & (n_earlier > 0)
    prev_t = np.where(ok, ts[np.maximum(pos - 1, 0)], -1)
    gap = (tt - prev_t) / 86400.0
    first_t = np.where(ok, ts[lo], -1)
    gap_old = (tt - first_t) / 86400.0
    cap = H == 256
    print(f"{s}: feeds {n}; user0 {(uu == 0).mean():.4f}; capped {cap.mean():.4f}; >=1 strictly earlier snapshot (uid!=0) {ok.mean():.4f} "
          f"(capped {(ok & cap).mean():.4f}, uncapped {(ok & ~cap).mean():.4f}); share of capped with earlier {(ok & cap).sum() / cap.sum():.4f}")
    q = np.percentile(gap[ok], [5, 25, 50, 75, 95])
    print(f"   gap to most recent earlier (days) 5/25/50/75/95: {' / '.join(f'{x:.2f}' for x in q)}")
    for k in (3, 5, 7, 10):
        has = ok & (gap_old >= k)
        print(f"   has an earlier snapshot >= {k} d back: {has.mean():.4f} of feeds (capped {(has & cap).mean():.4f})")
    # same-timestamp other snapshots of the same user (not strictly earlier): count
    pos_r = np.searchsorted(key, k_own, side="right")
    same = pos_r - pos - 1
    print(f"   feeds with another feed of the same user at the SAME timestamp (excluded as not strictly earlier): {((same > 0) & (uu != 0)).mean():.4f}")
