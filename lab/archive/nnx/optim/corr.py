"""Mean per-feed Pearson correlation between val predictions (ensemble-diversity diagnostic for C5 / C2).
Usage: python corr.py A.npy B.npy [C.npy ...]  (prints the pairwise matrix)"""
import sys
import numpy as np, polars as pl
FL = "<repo>/data/flat"
M = pl.read_parquet(f"{FL}/val/feeds.parquet")["M"].to_numpy().astype(np.int64)
fid = np.repeat(np.arange(len(M)), M)
def z(s):
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]
P = [z(np.load(f)) for f in sys.argv[1:]]
for i, a in enumerate(sys.argv[1:]):
    print(" ".join(f"{(np.bincount(fid, P[i] * P[j], len(M)) / M).mean():.4f}" for j in range(len(P))), a.split("/")[-1])
