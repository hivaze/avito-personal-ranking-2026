"""Synthesis check (CPU): M-split blend (TO-only for M<50, 0.5/0.5 FT/TO for M>=50) on the existing pool arrays,
val overall / M slices / coldpos-warmpos-free private-mix reweighting (src/pmix.py weights, report only).
Pools: preds/ens_<TAG>_{ft,to}_val.npy. Paired per-feed SE.
"""
import sys

import numpy as np
import polars as pl

ROOT = "<repo>"
sys.path.insert(0, f"{ROOT}/src")
from metrics import ndcg_per_feed  # noqa: E402
from pmix import weights, wmean_se  # noqa: E402

f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
n = len(M)
fid = np.repeat(np.arange(n), M)
W = weights()


def z(s):
    s = np.asarray(s, np.float64)
    mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


small = M < 50
for tag in ("G3", "PF7x", "M2", "Cd"):
    try:
        ft, to = z(np.load(f"{ROOT}/preds/ens_{tag}_ft_val.npy")), z(np.load(f"{ROOT}/preds/ens_{tag}_to_val.npy"))
    except FileNotFoundError:
        continue
    fin = 0.5 * ft + 0.5 * to
    ms = np.where(small[fid], to, fin)
    a, b, c = (ndcg_per_feed(s, y, off, M) for s in (fin, ms, to))
    d = b - a
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))  # noqa: E731
    pm, pse = wmean_se(d, W["private"])
    pmf, pmfse = wmean_se(a - c, W["private"])
    print(f"{tag:5s} final {a.mean():.5f}  M-split - final: all {d.mean():+.5f}+-{se(d):.5f}  "
          f"M<50 {d[small].mean():+.5f}+-{se(d[small]):.5f}  private-mix {pm:+.5f}+-{pse:.5f}  | "
          f"FT half (final - TO) val {np.mean(a - c):+.5f}  private-mix {pmf:+.5f}+-{pmfse:.5f}")
