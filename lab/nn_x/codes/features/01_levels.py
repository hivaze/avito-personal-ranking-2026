"""Value distributions of f1..f5: discrete levels, grids, floors, uniformity (quantile coding), per-feed level counts,
train/val/test parity. Train = uniform 200K-feed sample (seed 0)."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *

res = {}
for split in ["train", "val", "test"]:
    fidx = train_sample("uniform", 200_000, 0) if split == "train" else None
    d = load(split, fidx)
    n = len(d["f1"])
    print(f"\n===== {split}: feeds {d['n_feeds']}, rows {n}")
    for k in FS:
        v = d[k].astype(np.float64)
        u, c = np.unique(v, return_counts=True)
        nan = np.isnan(v).mean()
        top = np.argsort(-c)[:5]
        # grid search
        best = []
        for N in list(range(2, 257)):
            on = np.abs(u * N - np.round(u * N)) < 1e-4
            best.append((np.sum(c[on]) / n, N))
        best.sort(reverse=True)
        hist = np.histogram(v, bins=10, range=(0, 1))[0] / n
        # per-feed distinct levels and dominant-level share
        o = np.lexsort((v, d["feed"]))
        vs = v[o]; fs = d["feed"][o]
        newv = np.r_[True, (fs[1:] != fs[:-1]) | (vs[1:] != vs[:-1])]
        nlev = np.bincount(fs, newv)
        runid = np.cumsum(newv) - 1
        runlen = np.bincount(runid)
        runfeed = fs[newv]
        dom = np.zeros(d["n_feeds"]); np.maximum.at(dom, runfeed, runlen)
        Mf = np.bincount(d["feed"])
        print(f"{k}: nunique {len(u)} nan {nan:.4f} min {u.min():.5f} ({c[0]/n:.4f}) max {u.max():.5f} ({c[-1]/n:.4f}) mean {np.nanmean(v):.4f} sd {np.nanstd(v):.4f}")
        print(f"   top values: " + ", ".join(f"{u[i]:.5f}:{c[i]/n:.4f}" for i in top))
        print(f"   grid fit (share of rows exactly on k/N): " + ", ".join(f"N={N}:{s:.3f}" for s, N in best[:3]))
        print(f"   decile mass: " + " ".join(f"{h:.3f}" for h in hist))
        print(f"   per feed: distinct levels median {np.median(nlev):.0f} mean {nlev.mean():.1f}; distinct/M mean {np.mean(nlev/Mf):.3f}; dominant-level share mean {np.mean(dom/Mf):.3f} (M>=50: {np.mean((dom/Mf)[Mf>=50]):.3f})")
        res[(split, k)] = dict(nunique=len(u), mean=float(np.nanmean(v)), floor=float(c[0] / n))
    # train by traffic type
    if split == "train":
        p = d["p_random"]; M = d["M"]
        for lab, sel in [("random-like P>0.9 M>=50", (p > 0.9) & (M >= 50)), ("production P<0.05 M>=50", (p < 0.05) & (M >= 50))]:
            print(f"  train {lab}: rows {sel.sum()} means " + " ".join(f"{k} {d[k][sel].mean():.4f}" for k in FS))
