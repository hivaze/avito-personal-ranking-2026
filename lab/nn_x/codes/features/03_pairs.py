"""Item-pair equality test: is the tie structure of f1 / f2 / f3 / f5 inside a feed a FIXED partition of items
(a hidden item key such as micro-category or location) that repeats across feeds and users?
For item pairs (i, j) that co-occur in >= 2 feeds, measure the share of feeds in which f(i) == f(j) (non-floor, non-ceiling).
A hidden item key gives a bimodal share (always / never equal); per-user noise gives intermediate shares.
Train uniform sample 400K feeds (seed 1); items with >= 20 appearances; <= 30 such rows per feed (random)."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *

fidx = train_sample("uniform", 400_000, 1)
d = load("train", fidx)
item = d["item"].astype(np.int64)
u, inv, cnt = np.unique(item, return_inverse=True, return_counts=True)
keep = cnt[inv] >= 20
print(f"rows {len(item)}, rows with item seen >=20 times {keep.mean():.3f}, such items {np.sum(cnt>=20)}")
rng = np.random.default_rng(0)
idx = np.flatnonzero(keep)
# cap 30 per feed
feed = d["feed"][idx]
r = rng.random(len(idx))
o = np.lexsort((r, feed)); idx = idx[o]; feed = feed[o]
fstart = np.r_[True, feed[1:] != feed[:-1]]
rank = np.arange(len(idx)) - np.flatnonzero(fstart)[np.cumsum(fstart) - 1]
idx = idx[rank < 30]; feed = feed[rank < 30]
fstart = np.r_[True, feed[1:] != feed[:-1]]
starts = np.flatnonzero(fstart); lens = np.diff(np.r_[starts, len(idx)])
print(f"feeds with >=2 kept rows {np.sum(lens>=2)}, kept rows {len(idx)}")
A = []; B = []
for k in range(2, 31):
    fs = starts[lens == k]
    if len(fs) == 0: continue
    ti, tj = np.triu_indices(k, 1)
    A.append((fs[:, None] + ti[None, :]).ravel()); B.append((fs[:, None] + tj[None, :]).ravel())
A = idx[np.concatenate(A)]; B = idx[np.concatenate(B)]
print(f"pairs {len(A)}")
ia = item[A]; ib = item[B]
sw = ia > ib
ia2 = np.where(sw, ib, ia); ib2 = np.where(sw, ia, ib)
key = ia2 * (item.max() + 1) + ib2
ukey, kinv, kcnt = np.unique(key, return_inverse=True, return_counts=True)
multi = kcnt[kinv] >= 3
print(f"pairs co-occurring in >=3 feeds: {np.sum(kcnt>=3)} pair keys, {multi.sum()} pair-feed rows")
samecat = d["cat"][A] == d["cat"][B]
sameuser = None
floors = {"f1": (1/56, 55/56), "f2": (0.0, 63/64), "f3": (1/62, 61/62), "f5": (1/62, 61/62), "f4": (1/64, 63/64)}
for k in ["f3", "f1", "f2", "f5", "f4"]:
    v = d[k]
    va = v[A]; vb = v[B]
    lo, hi = floors[k]
    special = (np.abs(va - lo) < 1e-6) | (np.abs(va - hi) < 1e-6) | (va == 0)
    eq = (va == vb) & ~special
    valid = ~(((va == vb) & special))  # drop floor/ceiling ties
    sel = multi & valid
    # per pair-key share of equal over valid co-occurrences
    kk = kinv[sel]
    n_ = np.bincount(kk, minlength=len(ukey)); e_ = np.bincount(kk, eq[sel], minlength=len(ukey))
    sc = np.bincount(kinv, samecat, minlength=len(ukey)) / kcnt
    ok = n_ >= 3
    share = e_[ok] / n_[ok]
    print(f"\n{k}: pair-feed rows equal (excl. floor/ceiling ties) {eq[sel].mean():.4f}; pair keys with >=3 valid co-occurrences {ok.sum()}")
    for lab, m in [("same category", sc[ok] == 1), ("different category", sc[ok] == 0)]:
        s = share[m]
        if len(s) == 0: continue
        h = np.histogram(s, bins=[0, 1e-9, 0.2, 0.4, 0.6, 0.8, 1 - 1e-9, 1.01])[0] / len(s)
        print(f"  {lab}: n {len(s)}; share-equal histogram [0 | (0,.2) | .2-.4 | .4-.6 | .6-.8 | (.8,1) | 1]: " + " ".join(f"{x:.3f}" for x in h))
    # conditional consistency: given equal in a random first co-occurrence, P(equal in the others)
    np.save(OUT / f"pairs_{k}_share.npy", np.c_[e_[ok], n_[ok], sc[ok]])
