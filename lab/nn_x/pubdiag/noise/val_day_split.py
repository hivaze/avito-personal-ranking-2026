"""Part 5 (extra): val deltas by val day (Tue 20718 vs Thu 20720). If the m02 gain decayed with distance from the
train period, Thu < Tue would show it; public is 2-3 days after Thu, private 3-5 days."""
import sys
sys.path.insert(0, "<repo>/src")
from common import *  # noqa
from metrics import ndcg_per_feed
O = Path("<repo>/nn_x/pubdiag/noise")
f, M, o = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
day = (f["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
tue = day < 20720
nd = {k: ndcg_per_feed(np.load(O / f"cache_{k}_val.npy"), y, o, M) for k in ["Cd", "PF7x", "G3", "M2", "Pall", "G3c", "M2c"]}
fmt = lambda d: f"{d.mean():+.5f}±{d.std() / np.sqrt(len(d)):.5f}"
print(f"val feeds Tue {tue.sum()}  Thu {(~tue).sum()}")
print(f"{'contrast':11s} {'Tue':>18s} {'Thu':>18s} {'Thu-Tue':>18s}")
for a, b in [("PF7x", "Cd"), ("G3", "Cd"), ("M2", "Cd"), ("M2", "PF7x"), ("PF7x", "Pall"), ("G3", "G3c"), ("M2", "M2c")]:
    d = nd[a] - nd[b]
    dd = d[~tue].mean() - d[tue].mean(); sd = np.sqrt(d[tue].var() / tue.sum() + d[~tue].var() / (~tue).sum())
    print(f"{a + '-' + b:11s} {fmt(d[tue]):>18s} {fmt(d[~tue]):>18s} {dd:+.5f}±{sd:.5f}")
