"""Details for f2 (region affinity) and f1 (finer affinity).
f2: dominant-level value distribution across feeds (fixed code -> distance-like; varying -> affinity), is the dominant level the
feed max, is f2 == 0 an item property, target rate by f2 class (dominant / other non-zero / zero) on val and random-like train,
dependence on history length H.
f1: for same-category frequent pairs with MIXED equality, are the equal occurrences the low-affinity (backoff) ones?"""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from pairlib import *

def f2_classes(d):
    feed = d["feed"]; v = d["f2"]
    df = pl.DataFrame({"feed": feed, "f2": v})
    dom = (df.filter(pl.col("f2") > 0).group_by(["feed", "f2"]).agg(pl.len().alias("n"))
           .sort(["feed", "n", "f2"], descending=[False, True, True]).group_by("feed", maintain_order=True).first())
    domv = np.full(d["n_feeds"], np.nan); domv[dom["feed"].to_numpy()] = dom["f2"].to_numpy()
    mx = pl.DataFrame({"feed": feed, "f2": v}).group_by("feed").agg(pl.col("f2").max()).sort("feed")["f2"].to_numpy()
    cls = np.where(v == 0, 0, np.where(v == domv[feed], 2, 1))  # 0 zero, 1 other, 2 dominant
    return cls, domv, mx

for split, kind in [("val", None), ("train", "rl"), ("train", "prod")]:
    fidx = None if kind is None else train_sample(kind, 60_000, 2)
    d = load(split, fidx)
    cls, domv, mx = f2_classes(d)
    y = d["target"]; feed = d["feed"]
    nf = d["n_feeds"]
    has = ~np.isnan(domv)
    print(f"\n===== {split} {kind or ''}: feeds {nf}")
    q = np.nanquantile(domv, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"f2 dominant level value across feeds: quantiles 5/25/50/75/95% " + " ".join(f"{x:.3f}" for x in q) + f"; distinct dominant values {len(np.unique(domv[has]))}; share = 63/64 ceiling {np.mean(np.abs(domv[has]-63/64)<1e-6):.3f}")
    print(f"dominant level == feed max f2: {np.mean(domv[has] == mx[has]):.3f}")
    H = np.bincount(feed, d["H"]) / np.bincount(feed)
    for lo, hi in [(0, 1), (1, 50), (50, 150), (150, 256), (256, 257)]:
        m = has & (H >= lo) & (H < hi)
        if m.sum() > 50:
            print(f"   H in [{lo},{hi}): feeds {m.sum()}, median dominant f2 {np.median(domv[m]):.3f}")
    for c, lab in [(2, "dominant level"), (1, "other non-zero"), (0, "f2 == 0")]:
        m = cls == c
        print(f"   class {lab}: row share {m.mean():.3f}, target rate {y[m].mean()*100:.3f}%")
    # within-feed: conditional logit-free comparison: per-feed rate ratio dominant vs other (feeds having both and >=1 positive)
    # feed-level paired: mean over feeds of (rate_dom - rate_other)
    ok = np.ones(nf, bool)
    rd = np.bincount(feed, (cls == 2) * y, minlength=nf) / np.maximum(np.bincount(feed, cls == 2, minlength=nf), 1)
    ro = np.bincount(feed, (cls == 1) * y, minlength=nf) / np.maximum(np.bincount(feed, cls == 1, minlength=nf), 1)
    rz = np.bincount(feed, (cls == 0) * y, minlength=nf) / np.maximum(np.bincount(feed, cls == 0, minlength=nf), 1)
    both = (np.bincount(feed, cls == 2, minlength=nf) > 0) & (np.bincount(feed, cls == 1, minlength=nf) > 0) & (np.bincount(feed, y, minlength=nf) > 0)
    diff = rd[both] - ro[both]
    print(f"   within-feed rate(dominant) - rate(other non-zero): {diff.mean()*100:+.3f} pp (se {diff.std()/np.sqrt(both.sum())*100:.3f}, feeds {both.sum()})")
    bz = (np.bincount(feed, cls == 2, minlength=nf) > 0) & (np.bincount(feed, cls == 0, minlength=nf) > 0) & (np.bincount(feed, y, minlength=nf) > 0)
    dz = rd[bz] - rz[bz]
    print(f"   within-feed rate(dominant) - rate(f2==0): {dz.mean()*100:+.3f} pp (se {dz.std()/np.sqrt(bz.sum())*100:.3f}, feeds {bz.sum()})")
    if split == "train":
        pos = d["pos"].astype(np.float64)
        # within-feed relative position by class
        pr = np.empty(len(pos))
        o = np.lexsort((pos, feed)); fs = np.r_[True, feed[o][1:] != feed[o][:-1]]
        rk = np.arange(len(o)) - np.flatnonzero(fs)[np.cumsum(fs) - 1]
        pr[o] = rk / np.maximum(np.bincount(feed)[feed[o]] - 1, 1)
        print("   mean within-feed relative position (0=top): " + ", ".join(f"{lab} {pr[cls==c].mean():.3f}" for c, lab in [(2, 'dominant'), (1, 'other'), (0, 'zero')]))

# is f2 == 0 an item property? items with >= 5 appearances in a 400K train sample
fidx = train_sample("uniform", 400_000, 1)
d = load("train", fidx)
item = d["item"].astype(np.int64)
u, inv, cnt = np.unique(item, return_inverse=True, return_counts=True)
z = np.bincount(inv, d["f2"] == 0)
ok = cnt >= 5
fr = z[ok] / cnt[ok]
print(f"\nf2==0 per item (items with >=5 appearances, n={ok.sum()}): share of items never 0 {np.mean(fr==0):.3f}, always 0 {np.mean(fr==1):.3f}, mixed {np.mean((fr>0)&(fr<1)):.3f}")
# f1 mixed pairs: backoff test
d2, A, B, ukey, kinv, kcnt = build_pairs()
nk = len(ukey)
e_, n_ = pair_share(d2, A, B, kinv, nk, "f1")
samecat = np.bincount(kinv, d2["cat"][A] == d2["cat"][B], minlength=nk) / kcnt
mixed = (n_ >= 3) & (e_ > 0) & (e_ < n_) & (samecat == 1)
alw = (n_ >= 3) & (e_ == n_) & (samecat == 1)
row_m = mixed[kinv]
va = d2["f1"][A]; vb = d2["f1"][B]
lo, hi = FLOORS["f1"]
special = (np.abs(va - lo) < 1e-6) | (np.abs(va - hi) < 1e-6)
eq = (va == vb) & ~special
valid = ~((va == vb) & special)
m = row_m & valid
f3a = d2["f3"][A]
print(f"\nf1 mixed same-category pairs: {mixed.sum()} keys; pair-feed rows {m.sum()}")
print(f"   when equal: mean f1 {va[m & eq].mean():.3f}, mean f3 {f3a[m & eq].mean():.3f}, rows {np.sum(m & eq)}")
print(f"   when unequal: mean f1 {(0.5*(va+vb))[m & ~eq].mean():.3f}, mean f3 {f3a[m & ~eq].mean():.3f}, mean |diff| {np.abs(va-vb)[m & ~eq].mean():.4f}, rows {np.sum(m & ~eq)}")
ra = alw[kinv] & valid
print(f"   always-equal same-category pairs: mean f1 {va[ra].mean():.3f}, mean f3 {f3a[ra].mean():.3f}")
nv = (n_ >= 3) & (e_ == 0) & (samecat == 1)
rn = nv[kinv] & valid
print(f"   never-equal same-category pairs: mean f1 {(0.5*(va+vb))[rn].mean():.3f}, mean |diff| {np.abs(va-vb)[rn].mean():.4f}, mean f3 {f3a[rn].mean():.3f}")
# same user consistency of f1 equality for mixed keys
user = d2["user_id"][A]
