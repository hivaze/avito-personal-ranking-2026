"""Per-day medians / q10 / q90 / NaN share of features over val (Tue, Thu) and test (Sat..Tue), MSK days.
Usage: python drift_days.py name1,name2,...
"""
import sys
sys.path.insert(0, "<repo>/src")
from common import *

names = sys.argv[1].split(",")
cols = []
for split in ["val", "test"]:
    f, M, off = feed_meta(split)
    fidx = np.load(FLAT / split / "cand_fidx.npy")
    day = ((f["timestamp"].to_numpy().astype(np.int64) + 3 * 3600) // 86400)[fidx]
    for d in np.unique(day):
        m = day == d
        if m.sum() < 10000:
            continue
        cols.append((split, d, m))
lab = {20718: "Tue22", 20720: "Thu24", 20722: "Sat26", 20723: "Sun27", 20724: "Mon28", 20725: "Tue29"}
print("feature".ljust(22), " ".join(f"{lab.get(d, d):>17s}" for _, d, _ in cols))
for n in names:
    vals = {s: np.load(FEATS / s / f"{n}.npy") for s in ["val", "test"]}
    row = []
    for s, d, m in cols:
        x = vals[s][m]
        q = np.nanquantile(x, [0.1, 0.5, 0.9])
        row.append(f"{q[1]:7.3g} [{q[0]:.2g},{q[2]:.2g}]")
    print(n[4:].ljust(22), " ".join(f"{r:>17s}" for r in row))
