"""T3: how much of the train stream is dead weight for the listwise loss (feeds with 0 positives), by M bucket, and
how large is the M<5 share. Also the share of M<50 feeds (where p_random is forced to 0 -> treated as production,
excluded from the pool and the holdout)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "12")
import sys, numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
f, M, off = feed_meta("train")
y = np.load(FLAT / "train/cand_target.npy")
npos = np.add.reduceat(y.astype(np.int64), off)
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"]
out = []
def say(s): print(s, flush=True); out.append(s)
say(f"train feeds {len(M)}, candidates {len(y)}, feeds with 0 positives {np.mean(npos == 0):.4f} (candidate share {M[npos == 0].sum() / M.sum():.4f})")
say(f"train feeds with M<5: {np.mean(M < 5):.4f}; M<50: {np.mean(M < 50):.4f} (candidate share {M[M < 50].sum() / M.sum():.4f}); M<50 with >=1 pos: {np.mean((M < 50) & (npos > 0)):.4f}")
edges = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 301]
say("M bucket | feeds | share | P(npos=0) | mean npos | mean npos|npos>0 | candidate share")
for a, b in zip(edges[:-1], edges[1:]):
    s = (M >= a) & (M < b)
    say(f"[{a},{b}) | {s.sum()} | {s.mean():.3f} | {np.mean(npos[s] == 0):.3f} | {npos[s].mean():.3f} | {npos[s][npos[s] > 0].mean():.3f} | {M[s].sum() / M.sum():.3f}")
# the effective number of loss-contributing feeds per batch, by Mpad bucket (cand_budget 32768, max_feeds 512)
say("effective contributing feeds per batch (B = min(512, 32768 // Mpad)), by Mpad bucket:")
Mp = (M + 15) // 16 * 16
for mp in sorted(np.unique(Mp)):
    s = Mp == mp; B = int(max(1, min(512, 32768 // mp)))
    say(f"  Mpad {mp:3d}: feeds {s.sum():7d}, B {B:3d}, contributing {B * np.mean(npos[s] > 0):6.1f}, contributing share of batch {np.mean(npos[s] > 0):.3f}")
# p_random availability
nanp = np.isnan(pr)
say(f"p_random: NaN share {nanp.mean():.4f}; among M>=50: NaN {nanp[M >= 50].mean():.4f}, p>0.5 {np.nanmean(pr[M >= 50] > 0.5):.4f}, p>0.9 {np.nanmean(pr[M >= 50] > 0.9):.4f}")
say(f"val-like population in train (M>=50, p>0.5, npos>0): {((M >= 50) & (np.nan_to_num(pr) > 0.5) & (npos > 0)).sum()} feeds")
open("<repo>/nn_x/audit/weakspots/t3_train_npos.txt", "w").write("\n".join(out) + "\n")
