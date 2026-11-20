"""Step 4b: the one rank-local bias in the novelty table: new-to-user-category items at model ranks 1-3 are clicked
less than the softmax says (A/E 0.83-0.94 on both halves). Cross-fitted NDCG grid for a head demotion:
score - b * nov8 * 1[rank <= k] (k in 1, 3, 5), best (k, b) on one half applied to the other; plus the same for the
hmean5 top decile promoted inside the head (A/E 1.05-1.13 at ranks 1-10)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *  # noqa: F401,F403

F, M, off = layout()
half = np.load(HERE / "data/half.npy")
fid = np.repeat(np.arange(len(M)), M)
D = load_table(["score", "label", "rank", "cat_newest_idx", "hmean5", "gap_top"])
y = D["label"].to_numpy().astype(np.float64)
s = D["score"].to_numpy().astype(np.float64)
rk = D["rank"].to_numpy()
cni = D["cat_newest_idx"].to_numpy()
nov8 = ((cni < 0) | (cni >= 8)).astype(float)
hm = D["hmean5"].to_numpy().astype(np.float64)
hq = (hm >= np.nanquantile(hm, 0.9)).astype(float)
nd0 = ndcg_per_feed(s, y, off, M)
for name, x, sign in [("demote nov8 in head", nov8, -1), ("promote hmean5 top decile in head", hq, +1)]:
    cands = [(k, b) for k in (1, 3, 5, 10) for b in (0.05, 0.1, 0.2, 0.3, 0.5)]
    curves = np.array([ndcg_per_feed(s + sign * b * x * (rk <= k), y, off, M) - nd0 for k, b in cands])
    mA, mB = curves[:, half == 0].mean(1), curves[:, half == 1].mean(1)
    iA, iB = mA.argmax(), mB.argmax()
    d = np.where(half == 1, curves[iA], curves[iB])
    report_delta(d + nd0, nd0, M, half, f"{name}: A->{cands[iA]} B->{cands[iB]}")
    print(f"      in-sample best A {mA.max():+.5f} B {mB.max():+.5f}; full-val curve max {curves.mean(1).max():+.5f} at {cands[curves.mean(1).argmax()]}")
