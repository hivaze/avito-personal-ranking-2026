"""V5: are the 'rank-blend with f1' gains in eda_history_leak inflated by breaking f1's ties against a pessimistic f1 baseline?
Recompute with both baselines for co-visitation, train-history popularity, ALS profile, attention score."""
import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

FLAT = Path("<repo>/data/flat"); FE = Path("<repo>/data/feats/val")
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy()
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
fid = np.repeat(np.arange(len(M)), M)
rng = np.random.default_rng(1)


def prank(s, noise):
    s = np.nan_to_num(np.asarray(s, np.float64), nan=-1e18)
    o = np.lexsort((noise, s, fid))
    r = np.empty(len(s)); r[o] = np.arange(len(s)) - np.repeat(off, M)
    return r / np.repeat(M - 1, M)


f1 = np.load(FLAT / "val/cand_f1.npy").astype(np.float64)
pess = ndcg_per_feed(f1, y, off, M)
fair = np.mean([ndcg_per_feed(f1 + 1e-9 * rng.random(len(f1)), y, off, M) for _ in range(5)], axis=0)
print(f"f1 pessimistic {pess.mean():.4f}, f1 random tie-break (5 seeds) {fair.mean():.4f}")
for nm in ["seq_cv_invd20", "it_hist_et1", "it_hist_all", "als_exp5", "attn32"]:
    p = FE / f"{nm}.npy"
    if not p.exists(): print("missing", nm); continue
    x = np.load(p).astype(np.float64)
    for w in [0.25, 1.0]:
        res = []
        for k in range(5):
            noise = rng.random(len(y))
            # mid-rank style: f1 ties broken by noise inside prank -> this is what a naive rank blend does
            s = prank(f1, noise) + w * prank(x, rng.random(len(y)))
            res.append(ndcg_per_feed(s + 1e-12 * rng.random(len(y)), y, off, M))
        b = np.mean(res, axis=0)
        d1 = b - pess; d2 = b - fair
        print(f"  rank(f1)+{w}*rank({nm}): {b.mean():.4f}  vs pess f1 {d1.mean():+.4f}  vs fair f1 {d2.mean():+.4f}+-{d2.std()/np.sqrt(len(d2)):.4f}")
