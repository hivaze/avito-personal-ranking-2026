"""Reviewer: A/A of the re-cool harness (ref2 - ref, @last, val) with own code."""
import numpy as np, polars as pl, os, itertools
R = "<repo>"; D = f"{R}/nn_x/w5/preds"
f = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); y = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)
def nd(s):
    n = len(M); fid = np.repeat(np.arange(n), M); off = np.concatenate([[0], np.cumsum(M)[:-1]])
    o = np.lexsort((y, -s.astype(np.float64), fid)); rk = np.arange(len(s)) - np.repeat(off, M); d = 1.0 / np.log2(rk + 2.0)
    dcg = np.bincount(fid, weights=y[o] * d, minlength=n); npos = np.bincount(fid, weights=y, minlength=n)
    return dcg / np.bincount(fid, weights=(rk < np.repeat(npos, M)) * d, minlength=n)
def z(s):
    fid = np.repeat(np.arange(len(M)), M); mu = np.bincount(fid, weights=s) / M; sd = np.sqrt(np.maximum(np.bincount(fid, weights=(s - mu[fid]) ** 2) / M, 1e-12)); return (s - mu[fid]) / sd[fid]
pairs = [("k", 951), ("k", 953), ("k", 955), ("k", 957), ("l", 951), ("l", 953)]
Za, Zb, d = [], [], []
for st, s in pairs:
    a = np.load(f"{D}/nn_w5_{st}cd_ref_s{s}@last_trainonly_val.npy").astype(np.float64); b = np.load(f"{D}/nn_w5_{st}cd_ref2_s{s}@last_trainonly_val.npy").astype(np.float64)
    x = nd(b) - nd(a); d.append(x.mean()); Za.append(z(a)); Zb.append(z(b))
    print(f"w5_{st}cd ref2 - ref s{s}: {x.mean():+.5f} +- {x.std(ddof=1) / np.sqrt(len(x)):.5f}")
print(f"member RMS {np.sqrt(np.mean(np.square(d))):.5f}")
Za, Zb = np.array(Za), np.array(Zb); ds = []
for bits in itertools.product([0, 1], repeat=5):
    sw = np.array((0,) + bits, bool)
    A = np.where(sw[:, None], Zb, Za).mean(0); B = np.where(sw[:, None], Za, Zb).mean(0)
    ds.append(nd(A).mean() - nd(B).mean())
ds = np.array(ds); s6 = np.sqrt((ds ** 2).mean())
print(f"pool level over the 32 within-pair label swaps: RMS {s6:.5f} (max {np.abs(ds).max():.5f}); x sqrt(6/16) = {s6 * np.sqrt(6 / 16):.5f}")
more = [(st, s) for st in "kl" for s in range(951, 959) if os.path.exists(f"{D}/nn_w5_{st}cd_ref2_s{s}@last_trainonly_val.npy") and (st, s) not in pairs]
print("further finished (ref, ref2) pairs since the study's read:", more)
