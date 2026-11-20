"""web_shift probe s9 (information only, after the s4 read): the s4 pool deltas reweighted to the public / private feed mix (src/pmix.py weights)."""
import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
import pmix
from metrics import ndcg_per_feed
ROOT = Path("<repo>"); HERE = ROOT / "reports/TRICKS_1115/web_shift_probe"
f = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
n = len(M); fid = np.repeat(np.arange(n), M); y = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
def z(s):
    s = s.astype(np.float64); mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]
V = [f"w5_e10_kd05w192_s{s}" for s in (951, 952, 953, 954)]
nd = {a: ndcg_per_feed(np.mean([z(np.load(HERE / "preds" / f"cpu_{v}_{a}.npy")) for v in V], 0), y, off, M) for a in ("base", "A", "B")}
W = pmix.weights()
for a in ("A", "B"):
    d = nd[a] - nd["base"]
    for k in ("public", "private"):
        w = np.asarray(W[k], float); w = w / w.sum(); mu = (w * d).sum(); se = np.sqrt((w ** 2 * (d - mu) ** 2).sum())
        print(f"arm {a} pmix-{k}: {mu:+.5f} +- {se:.5f} (z {mu/se:+.2f})")
