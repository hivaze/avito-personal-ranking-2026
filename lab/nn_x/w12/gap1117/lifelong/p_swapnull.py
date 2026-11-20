"""Calibration of the bucket read (CPU, read-only; labels of the two REFERENCE re-cool pools only, no extension arm exists):
the 16 seed pairs (ref_s, ref2_s) are exchangeable, so swapping the members of a random subset of pairs between the two pools
gives A/A contrasts with the same structure as an arm-vs-ref read. The spread of the bucket pooled z over N_SWAP random swap
patterns says whether the plain per-feed SE of read_hx.py is right (sd 1) or too small (sd > 1).
  python p_swapnull.py > out_p_swapnull.txt
"""
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, "<repo>/lab/src")
from metrics import ndcg_per_feed  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hx_lib import DATA_OUT, Flat  # noqa: E402

N_SWAP = 60
W = "<repo>/lab/nn_x/w5/preds"
F = Flat()
hold, _ = F.holdout()
fv = pl.read_parquet("<repo>/data/flat/val/feeds.parquet")
Mv, ov = fv["M"].to_numpy().astype(np.int64), fv["cand_off"].to_numpy().astype(np.int64)
yv = np.load("<repo>/data/flat/val/cand_target.npy").astype(np.float32)
Mh, oh = np.load(f"{W}/holdout/holdout_meta.npy").astype(np.int64)
yh = np.load(f"{W}/holdout/holdout_y.npy").astype(np.float32)
uv = fv["user_id"].to_numpy()
uh = F.uid["train"][hold]


def z(off, M, x):
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, weights=x, minlength=len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, weights=x ** 2, minlength=len(M)) / M - mu ** 2, 0)) + 1e-9
    return ((x - mu[fid]) / sd[fid]).astype(np.float32)


names = [(f"w5_{st}cd_ref_s{x}", f"w5_{st}cd_ref2_s{x}") for st in "kl" for x in range(951, 959)]
ZV = np.stack([[z(ov, Mv, np.load(f"{W}/nn_{v}_trainonly_val.npy").astype(np.float64)) for v in p] for p in names])  # [16,2,n]
ZH = np.stack([[z(oh, Mh, np.load(f"{W}/holdout/nn_{v}_holdout.npy").astype(np.float64)) for v in p] for p in names])
bv = (np.load(f"{DATA_OUT}/val/ext_n.npy") > 0) & np.load(f"{DATA_OUT}/val/ext_capped.npy")
bh = ((np.load(f"{DATA_OUT}/train/ext_n.npy") > 0) & np.load(f"{DATA_OUT}/train/ext_capped.npy"))[hold]
se = lambda d: float(d.std() / np.sqrt(len(d)))


def cse(d, u):
    """user-cluster-robust SE of a mean."""
    _, inv = np.unique(u, return_inverse=True)
    s = np.bincount(inv, weights=d - d.mean())
    return float(np.sqrt((s ** 2).sum()) / len(d))


rng = np.random.default_rng(1117)
rows = []
for t in range(N_SWAP + 1):
    sw = np.zeros(16, int) if t == 0 else rng.integers(0, 2, 16)
    a_v, b_v = ZV[np.arange(16), sw].mean(0), ZV[np.arange(16), 1 - sw].mean(0)
    a_h, b_h = ZH[np.arange(16), sw].mean(0), ZH[np.arange(16), 1 - sw].mean(0)
    dv = ndcg_per_feed(b_v, yv, ov, Mv) - ndcg_per_feed(a_v, yv, ov, Mv)
    dh = ndcg_per_feed(b_h, yh, oh, Mh) - ndcg_per_feed(a_h, yh, oh, Mh)
    zbv, zbh = dv[bv].mean() / se(dv[bv]), dh[bh].mean() / se(dh[bh])
    zav, zah = dv.mean() / se(dv), dh.mean() / se(dh)
    cz = (dv[bv].mean() / cse(dv[bv], uv[bv]) + dh[bh].mean() / cse(dh[bh], uh[bh])) / np.sqrt(2)
    rows.append([zbv, zbh, (zbv + zbh) / np.sqrt(2), zav, zah, (zav + zah) / np.sqrt(2), cz,
                 se(dv[bv]), se(dh[bh]), cse(dv[bv], uv[bv]), cse(dh[bh], uh[bh])])
    if t == 0:
        print(f"pattern 0 (= ref2 - ref): bucket z val {zbv:+.2f} holdout {zbh:+.2f} pooled {(zbv + zbh) / np.sqrt(2):+.2f}; "
              f"user-cluster SE val {cse(dv[bv], uv[bv]):.5f} (plain {se(dv[bv]):.5f}), holdout {cse(dh[bh], uh[bh]):.5f} (plain {se(dh[bh]):.5f})", flush=True)
r = np.array(rows[1:])
print(f"{N_SWAP} random swap patterns (A/A by construction):")
for j, nm in enumerate(("bucket z val", "bucket z holdout", "bucket pooled z", "all-feeds z val", "all-feeds z holdout", "all-feeds pooled z", "bucket pooled z (cluster SE)")):
    print(f"  {nm:30s}: mean {r[:, j].mean():+.2f}, sd {r[:, j].std():.2f}, share |z| >= 1.96 {(np.abs(r[:, j]) >= 1.96).mean():.3f}, share z >= 2.5 {(r[:, j] >= 2.5).mean():.3f}")
print(f"  mean SE val bucket plain {r[:, 7].mean():.5f} / cluster {r[:, 9].mean():.5f}; holdout bucket plain {r[:, 8].mean():.5f} / cluster {r[:, 10].mean():.5f}")
print("Note: swap patterns share members, so they are not independent draws; sd near 1 means the plain SE is calibrated.")
