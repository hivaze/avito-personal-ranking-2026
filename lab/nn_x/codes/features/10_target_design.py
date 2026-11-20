"""Target / feed design facts: share of feeds with >= 1 positive (train, val), positives per feed, target rate vs position
counted from the END of the shown list on train random-like and production feeds (feed truncation at the last interaction?)."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *

for name, split, kind in [("val", "val", None), ("train_uniform", "train", "uniform"), ("train_rl", "train", "rl"), ("train_prod", "train", "prod")]:
    fidx = None if kind is None else train_sample(kind, 100_000 if kind == "uniform" else 60_000, 4)
    d = load(split, fidx)
    y = d["target"].astype(np.float64); feed = d["feed"]; nf = d["n_feeds"]
    npos = np.bincount(feed, y, minlength=nf); M = np.bincount(feed, minlength=nf)
    print(f"\n{name}: feeds {nf}, target rate {y.mean()*100:.2f}%, feeds with 0 positives {np.mean(npos==0):.4f}, positives/feed mean {npos.mean():.2f} median {np.median(npos):.0f}, share with exactly 1 {np.mean(npos==1):.3f}, >=3 {np.mean(npos>=3):.3f}")
    if split == "train":
        pos = d["pos"].astype(np.int64)
        o = np.lexsort((pos, feed)); fs = np.r_[True, feed[o][1:] != feed[o][:-1]]
        rk = np.arange(len(o)) - np.flatnonzero(fs)[np.cumsum(fs) - 1]
        from_end = np.empty(len(o), np.int64); from_end[o] = M[feed[o]] - 1 - rk
        from_start = np.empty(len(o), np.int64); from_start[o] = rk
        m50 = M[feed] >= 50
        print("   rate by rank from END (0 = last shown), M>=50: " + " ".join(f"{k}:{y[(from_end==k)&m50].mean()*100:.1f}%" for k in [0, 1, 2, 3, 5, 10, 20]))
        print("   rate by rank from START, M>=50: " + " ".join(f"{k}:{y[(from_start==k)&m50].mean()*100:.1f}%" for k in [0, 1, 2, 3, 5, 10, 20]))
        last_pos = np.zeros(nf); np.maximum.at(last_pos, feed, np.where(y > 0, from_start, -1))
        has = npos > 0
        print(f"   feeds with >=1 positive: the LAST positive is the last shown item in {np.mean((M[has]-1-last_pos[has])==0):.3f}; within the last 3 items {np.mean((M[has]-1-last_pos[has])<=2):.3f}")
