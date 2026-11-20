"""critic hz_unread: BLOCK 6 (information). Train-only val / pmix read of the unread hz-family re-cools. Light CPU."""
import os, sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, polars as pl
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed
import pmix
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); coff = f["cand_off"].to_numpy().astype(np.int64); nF = len(M); fid = np.repeat(np.arange(nF), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
def zf(x):
    mu = np.add.reduceat(x, coff) / M; d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, coff) / M); return d / np.maximum(sd[fid], 1e-9)
def pool(paths): return zf(np.mean([zf(np.load(p).astype(np.float64)) for p in paths], 0))
def arm(a): return pool([ROOT / "nn_x/w5/preds" / f"nn_w5_{h}_{a}_s{s}@last_trainonly_val.npy" for h in ("kcd", "lcd") for s in range(951, 959)])
P = pool([ROOT / "preds" / f"nn_abl_e10_pop1kd05w192_s{s}_trainonly_val.npy" for s in range(951, 955)])
W = pmix.weights(); short = M < 50
def wm(d, w):
    m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum(); return m, np.sqrt(np.average((d - m) ** 2, weights=w) / neff)
def rd(a, b):
    d = ndcg_per_feed(a, y, coff, M) - ndcg_per_feed(b, y, coff, M)
    mp, sp = wm(d, W["public"]); mr, sr = wm(d, W["private"])
    return (f"val {d.mean():+.5f}±{d.std() / np.sqrt(nF):.5f} | M<50 {d[short].mean():+.5f}±{d[short].std() / np.sqrt(short.sum()):.5f} | "
            f"pmix-public {mp:+.5f}±{sp:.5f} | pmix-private {mr:+.5f}±{sr:.5f} (z {mr / sr:+.2f})")
ref = arm("ref"); hz = arm("hz")
print(f"ref 16-pool val {ndcg_per_feed(ref, y, coff, M).mean():.5f} | hz {ndcg_per_feed(hz, y, coff, M).mean():.5f} | P {ndcg_per_feed(P, y, coff, M).mean():.5f}", flush=True)
for a in ("hz", "hzt", "hznFS", "hz2nF"):
    A = hz if a == "hz" else arm(a)
    print(f"[{a}] (a) arm - ref : {rd(A, ref)}")
    if a != "hz":
        print(f"[{a}] (a) arm - hz  : {rd(A, hz)}")
    print(f"[{a}] (b) admixture form, arm - ref: {rd(zf(0.8 * A + 0.2 * P), zf(0.8 * ref + 0.2 * P))}", flush=True)
