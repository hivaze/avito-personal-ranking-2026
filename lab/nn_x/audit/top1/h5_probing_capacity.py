"""H5: feasibility of leaderboard probing given 10-digit public scores and 5 submissions/day.
Uses VAL labels as a stand-in for the public test (same construction, 14,423 vs 8,562 feeds).
Computes (1) the per-feed score change of a single adjacent swap at ranks (r, r+1) when exactly one is positive,
(2) the number of distinguishable magnitudes at the 1e-10 public precision, i.e. how many feeds one submission can
probe at once (group testing by distinct magnitudes), (3) the public gain per feed whose top positive is learned.
"""
import sys
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path("<repo>"); sys.path.insert(0, str(ROOT / "src"))
from common import feed_meta  # noqa: E402
off, M = feed_meta("val")[:2] if False else (None, None)
f = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64)
y = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.int64)
off = np.concatenate([[0], np.cumsum(M)[:-1]]); n = len(M)
npos = np.add.reduceat(y, off); N_PUB = 8562
idcg = np.array([sum(1 / np.log2(i + 2) for i in range(min(k, 30))) if k > 0 else np.nan for k in range(0, 60)])
print(f"val feeds {n}; npos distribution: 0:{(npos==0).sum()} 1:{(npos==1).sum()} 2:{(npos==2).sum()} 3+:{(npos>=3).sum()}; mean {npos.mean():.3f}")
# (1) single adjacent swap deltas (one positive among the two): |1/log2(r+1) - 1/log2(r+2)| / IDCG(npos)
res = 1e-10 * N_PUB  # resolution on the SUM of per-feed NDCG over public feeds
print(f"public score precision 1e-10 -> resolution on the per-feed NDCG sum: {res:.2e}")
for r in (1, 2, 3, 5, 10):
    g = abs(1 / np.log2(r + 1) - 1 / np.log2(r + 2))
    d = g / idcg[np.clip(npos, 0, 59)]
    d = d[npos > 0]
    print(f"swap ranks ({r},{r+1}): per-feed delta mean {d.mean():.4f} min {d.min():.4f} max {d.max():.4f}; "
          f"distinct values (npos 1..30): {len(np.unique(np.round(d, 9)))}; ratio max delta / resolution {d.max()/res:.1e}")
# (2) capacity: with k feeds probed at once using swaps whose magnitudes form a superincreasing set, the sum is decodable
# as long as each magnitude exceeds the sum of all smaller ones. Deltas are limited to the menu {g_r / IDCG(npos)}; the
# IDCG is unknown (npos in 1..30 -> up to 30 values) so a naive decode needs magnitudes separated by > max IDCG ratio.
# Lower bound (conservative): probe one feed per distinct magnitude class with a 2x safety factor.
g = np.array([abs(1 / np.log2(r + 1) - 1 / np.log2(r + 2)) for r in range(1, 300)])
levels = np.log(g.max() / res) / np.log(2.0)   # superincreasing capacity in bits
print(f"bits per submission (superincreasing bound on the delta range / resolution): {levels:.1f}")
print("-> group testing could resolve on the order of 20-30 feeds' 'is one of these two a positive' bits per submission;")
print("   a yes/no per feed per submission needs ~log2(M) submissions per feed to locate ONE positive by bisection.")
# (3) gain per feed when the top positive is moved to rank 1 from G3's rank: use G3 val ranks of positives
import json
s = np.load(ROOT / "preds/ens_G3_final_val.npy").astype(np.float64)
from metrics import ndcg_per_feed  # noqa: E402
base = ndcg_per_feed(s, y.astype(np.float32), off, M)
s2 = s.copy()
for i in range(n):
    sl = slice(off[i], off[i] + M[i]); yy = y[sl]
    if yy.sum() == 0: continue
    j = np.argmax(np.where(yy > 0, s[sl], -np.inf)); s2[off[i] + j] = s[sl].max() + 1.0
one = ndcg_per_feed(s2, y.astype(np.float32), off, M)
gain = (one - base)
print(f"G3 val {base.mean():.5f}; moving ONE known positive per feed to the top: per-feed gain mean {gain[npos>0].mean():.4f} "
      f"(median {np.median(gain[npos>0]):.4f}); all feeds if done for every feed: {one.mean():.5f}")
for k in (50, 100, 300, 1000, 3000):
    print(f"  learning the top positive of {k} public feeds -> public gain ~ {k*gain[npos>0].mean()/N_PUB:+.4f}")
print("budget: 5 subs/day x 41 days (Oct 28 - Dec 8) = 205 subs per team; 4 accounts -> 820. At ~1 located positive per")
print("~log2(median M ~ 70) ~ 6 bisection probes per feed, 205 subs -> ~35 feeds (+0.0025 public), 820 -> ~140 (+0.010);")
print("with magnitude-coded group testing (20+ feeds per probe) the same budget locates thousands of positives (+0.1 or more).")
print("A probing team therefore would not stop at +0.0036 over #2 unless deliberately disguised; nothing transfers to private.")
