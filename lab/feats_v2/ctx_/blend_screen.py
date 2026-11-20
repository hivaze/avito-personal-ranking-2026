"""Cheap screening (no training): blend the cached base OOF (mean of z-scored seeds) with a feature, both within-feed z-scored,
score = z(base) + w * z(feature), w in a small grid (both signs). Reports the best delta vs base and its paired SE.
Optimistic (w picked on val) -> only for ranking candidates before the harness.
Usage: python blend_screen.py <prefix or comma list>
"""
import sys
sys.path.insert(0, "<repo>/src")
from common import *

arg = sys.argv[1]
names = arg.split(",") if "," in arg else sorted(p.stem for p in (FEATS / "val").glob(f"{arg}*.npy"))
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
fidx = np.load(FLAT / "val/cand_fidx.npy")
import os
_seeds = [p for p in sorted(Path("<repo>/data/eval_cache").glob("base_oof_s*.npy"))]
base = None


def fz(x):
    x = x.astype(np.float64)
    nan = np.isnan(x)
    cnt = np.bincount(fidx, weights=~nan, minlength=len(M))
    mu = np.bincount(fidx, weights=np.where(nan, 0, x), minlength=len(M)) / np.maximum(cnt, 1)
    x = np.where(nan, mu[fidx], x)
    d = x - mu[fidx]
    sd = np.sqrt(np.bincount(fidx, weights=d * d, minlength=len(M)) / M)
    return d / (sd[fidx] + 1e-9)


zb = np.mean([fz(np.load(p)) for p in _seeds], axis=0)  # mean of within-feed z-scored base OOFs over seeds
pb = ndcg_per_feed(zb, y, off, M)
print(f"base {pb.mean():.5f}")
for n in names:
    z = fz(np.load(FEATS / "val" / f"{n}.npy"))
    best = (0, 0.0, 0.0)
    for w in [-0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4]:
        d = ndcg_per_feed(zb + w * z, y, off, M) - pb
        if d.mean() > best[1]:
            best = (w, d.mean(), d.std() / np.sqrt(len(d)))
    print(f"{n:32s} w {best[0]:+.2f} delta {best[1]:+.5f} se {best[2]:.5f}", flush=True)
