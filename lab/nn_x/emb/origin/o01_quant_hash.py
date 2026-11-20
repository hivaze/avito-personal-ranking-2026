"""O1: quantisation fingerprint + exact-duplicate census of the item embedding tables.

- quantisation: ||v||^2 mean/sd vs the rounding-noise prediction for round(256 * unit vector);
  per-vector max|v| (absmax-scaling test); per-dim min/max (per-dim calibration test).
- duplicates: 2 x 64-bit row hashes of every row of each split table (train 81.6M rows, all of it),
  share of rows whose vector is bit-identical to another row (rows are distinct item_ids within a split).
Outputs: out/hash_{split}.npy (N,2) uint64, out/o01.json, log to stdout.
"""
import json, time, os
import numpy as np

T0 = time.time()
ROOT = "<repo>"
OUT = f"{ROOT}/nn_x/emb/origin/out"
os.makedirs(OUT, exist_ok=True)
N = {"train": 81557944, "val": 2516711, "test": 4710948}
rng = np.random.default_rng(0)
M1 = (rng.integers(1, 2**63, size=16, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1))
M2 = (rng.integers(1, 2**63, size=16, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1))


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def mix(h):
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xff51afd7ed558ccd)
    h ^= h >> np.uint64(33)
    return h


def row_hash(x8):
    u = x8.view(np.uint64)  # (n,16)
    h1 = np.zeros(len(u), np.uint64)
    h2 = np.zeros(len(u), np.uint64)
    with np.errstate(over="ignore"):
        for k in range(16):
            h1 += u[:, k] * M1[k]
            h2 += (u[:, k] ^ np.uint64(0x9e3779b97f4a7c15)) * M2[k]
        return mix(h1), mix(h2)


res = {}
for split in ["val", "test", "train"]:
    mm = np.memmap(f"{ROOT}/data/competition/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r",
                   shape=(N[split], 128))
    H = np.zeros((N[split], 2), np.uint64)
    CH = 4_000_000
    for s in range(0, N[split], CH):
        x = np.ascontiguousarray(mm[s:s + CH])
        H[s:s + len(x), 0], H[s:s + len(x), 1] = row_hash(x)
    np.save(f"{OUT}/hash_{split}.npy", H)
    log(split, "hashed")
    _, inv, cnt = np.unique(H[:, 0], return_inverse=True, return_counts=True)
    rc = cnt[inv]
    # verify with the second hash: rows sharing h1 must share h2 (else collision)
    o = np.lexsort((H[:, 1], H[:, 0]))
    h1s, h2s = H[o, 0], H[o, 1]
    same1 = h1s[1:] == h1s[:-1]
    coll = int((same1 & (h2s[1:] != h2s[:-1])).sum())
    r = {
        "rows": int(N[split]),
        "distinct_vectors": int(len(cnt)),
        "share_rows_with_identical_twin": float((rc > 1).mean()),
        "rows_in_groups_ge3": float((rc >= 3).mean()),
        "max_group": int(cnt.max()),
        "group_size_hist": {str(k): int(v) for k, v in zip(*np.unique(np.minimum(cnt[cnt > 1], 11), return_counts=True))},
        "h1_collisions_not_h2": coll,
    }
    # quantisation fingerprint on a sample
    ns = min(2_000_000, N[split])
    idx = np.sort(rng.choice(N[split], ns, replace=False)) if ns < N[split] else np.arange(ns)
    x = mm[idx].astype(np.int32)
    n2 = (x * x).sum(1).astype(np.float64)
    nrm = np.sqrt(n2)
    amax = np.abs(x).max(1)
    r.update({
        "sample_rows": int(ns),
        "norm2_mean": float(n2.mean()), "norm2_pred_round256": 65536 + 128 / 12,
        "norm2_pred_scale255": 255**2 + 128 / 12,
        "norm_mean": float(nrm.mean()), "norm_sd": float(nrm.std()), "norm_sd_pred_rounding": float(np.sqrt(1 / 12)),
        "norm_q": [float(v) for v in np.quantile(nrm, [0, .001, .01, .5, .99, .999, 1])],
        "share_zero_vectors": float((n2 == 0).mean()),
        "absmax_q": [float(v) for v in np.quantile(amax, [0, .01, .5, .99, 1])],
        "share_rows_hit_127": float((amax >= 127).mean()),
        "per_dim_min_range": [int(x.min(0).min()), int(x.min(0).max())],
        "per_dim_max_range": [int(x.max(0).min()), int(x.max(0).max())],
        "per_dim_distinct_values_median": float(np.median([len(np.unique(x[:200000, d])) for d in range(128)])),
        "value_min": int(x.min()), "value_max": int(x.max()),
        "share_exact_zero_values": float((x == 0).mean()),
    })
    # rounding test: fractional residual of the unit vector at scale 256 cannot be seen directly, but parity of
    # values should be uniform (no scale-2 artefacts) and the histogram of v mod 4 should be flat
    r["value_mod4_shares"] = [float(v) for v in np.bincount((x[:200000].ravel() % 4), minlength=4) / x[:200000].size]
    res[split] = r
    log(split, json.dumps(r))
    del H

json.dump(res, open(f"{OUT}/o01.json", "w"), indent=1)
log("done")
