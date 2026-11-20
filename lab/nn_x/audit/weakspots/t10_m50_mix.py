"""T10: train feeds with M<50 (51% of train feeds, 14% of candidates; val/test are 41%/39% M<50 feeds) get p_random := 0
in nn/train.py, i.e. they are labelled PRODUCTION for the domain token, receive the K9 production-order aux at full
weight (1-p = 1), are excluded from the cooldown pool and from the holdout. How many of them are actually random-order
(exploration) feeds?  Label-free proxy = gap share of pos (removed already-viewed items: ~2% in random-like vs ~8% in
production feeds for M>=50, eda_labels_position). Position bias check (label-based, descriptive): P(y | pos rank 0) / base
by gap-share bin, M<50 vs M>=50 random-like / production-like."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
f, M, off = feed_meta("train"); M = M.astype(np.int64)
y = np.load(FLAT / "train/cand_target.npy").astype(np.float32)
pos = np.load(FLAT / "train/cand_pos.npy").astype(np.int64)
pr = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"], nan=-1)
fid = np.repeat(np.arange(len(M)), M)
maxpos = np.maximum.reduceat(pos, off)
gap = 1 - M / (maxpos + 1)   # share of removed slots
# rank within feed by pos
o = np.lexsort((pos, fid)); rk = np.empty(len(pos), np.int64); rk[o] = np.arange(len(pos)) - np.repeat(off, M)
y0 = np.zeros(len(M)); y0[fid[rk == 0]] = y[rk == 0]   # target of the pos-rank-0 item per feed
base = np.bincount(fid, y) / M                           # per-feed positive rate
out = []
def say(s): print(s, flush=True); out.append(s)
say(f"train feeds {len(M)}; M<50 share {np.mean(M < 50):.3f}; gap share mean: M<50 {gap[M < 50].mean():.3f}, M>=50 p>0.9 {gap[(M >= 50) & (pr > 0.9)].mean():.3f}, M>=50 p<0.1 {gap[(M >= 50) & (pr >= 0) & (pr < 0.1)].mean():.3f}")
say("lift of the pos-rank-0 item (P(y|rank0)/mean rate) by gap-share bin:")
bins = [(-0.001, 0.0001), (0.0001, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1.01)]
for lo, hi in [(5, 20), (20, 50), (50, 300)]:
    row = []
    for a, b in bins:
        s = (M >= lo) & (M < hi) & (gap > a) & (gap <= b)
        if s.sum() < 500: row.append(f"gap({a:.2f},{b:.2f}] n<500"); continue
        row.append(f"gap({a:.2f},{b:.2f}] n={s.sum()} lift {y0[s].mean() / base[s].mean():.2f}")
    say(f"  M[{lo},{hi}): " + " | ".join(row))
for lo, hi in [(50, 300)]:
    s = (M >= lo) & (pr > 0.9); say(f"  reference M>=50 p_random>0.9: n={s.sum()} lift {y0[s].mean() / base[s].mean():.2f}; p_random<0.1: lift {y0[(M >= lo) & (pr >= 0) & (pr < 0.1)].mean() / base[(M >= lo) & (pr >= 0) & (pr < 0.1)].mean():.2f}")
# position curve for M<50 zero-gap feeds vs others: mean target by rank 0..9
for name, s in [("M<50 gap=0", (M < 50) & (gap <= 0.0001)), ("M<50 gap>0", (M < 50) & (gap > 0.0001)), ("M>=50 p>0.9", (M >= 50) & (pr > 0.9)), ("M>=50 p<0.1", (M >= 50) & (pr >= 0) & (pr < 0.1))]:
    rows = s[fid]
    curve = [y[rows & (rk == r)].mean() for r in range(8)]
    say(f"  {name:14s} feeds {s.sum():8d}: P(y|rank r) r=0..7: " + " ".join(f"{c:.4f}" for c in curve) + f"  mean rate {y[rows].mean():.4f}")
# share of M<50 feeds that look random by the gap criterion, and their share of the K9 aux mass (feeds with (1-p)=1)
s_rand = (M < 50) & (gap <= 0.0001)
say(f"M<50 feeds with zero gap: {s_rand.sum()} ({s_rand.mean() / np.mean(M < 50):.3f} of M<50 feeds); these carry the K9 aux at weight 1 (p_random forced to 0) and the production domain token")
open("<repo>/nn_x/audit/weakspots/t10_m50_mix.txt", "w").write("\n".join(out) + "\n")
