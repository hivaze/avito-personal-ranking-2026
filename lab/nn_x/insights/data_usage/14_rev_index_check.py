"""How the NN position tower indexed 'rank from end': rev = clamp(M-1-pos, 0) with the RAW shown position (nn/model.py
forward), vs the true within-feed rank-from-end (gaps = removed already-viewed items). CPU, read-only."""
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path("<repo>"); D = ROOT / "nn_x/insights/data_usage/out"
t = pl.read_parquet(D / "train_feeds.parquet", columns=["M", "cls", "gaps"])
M = t["M"].to_numpy().astype(np.int64); cls = t["cls"].to_numpy(); n = len(M)
fid = np.repeat(np.arange(n), M)
pos = np.load(ROOT / "data/flat/train/cand_pos.npy").astype(np.int64)
y = np.load(ROOT / "data/flat/train/cand_target.npy").astype(np.int8)
off = np.concatenate([[0], np.cumsum(M)[:-1]])
o = np.lexsort((pos, fid)); rank = np.empty(len(pos), np.int64); rank[o] = np.arange(len(pos)) - off[fid[o]]; del o
rk_true = M[fid] - 1 - rank
rev_nn = np.clip(M[fid] - 1 - pos, 0, None)
print(f"feeds with gaps: {np.mean(t['gaps'].to_numpy() > 0):.3f}; mean gaps {t['gaps'].mean():.2f}")
for nm, mk in [("prod M>=50", (cls[fid] == 0) & (M[fid] >= 50)), ("prod M<50", (cls[fid] == 0) & (M[fid] < 50)), ("random-like", cls[fid] >= 1)]:
    a = mk & (rev_nn == 0); b = mk & (rk_true == 0)
    print(f"{nm:12s}: NN rev index 0 holds {a.sum() / np.unique(fid[mk]).size:.2f} candidates/feed, click rate {y[a].mean():.4f}; "
          f"true last item click rate {y[b].mean():.4f}; NN rev 1 rate {y[mk & (rev_nn == 1)].mean():.4f} vs true rkend 1 {y[mk & (rk_true == 1)].mean():.4f}; "
          f"share of true last items with rev index 0: {np.mean(rev_nn[b] == 0):.3f}")
