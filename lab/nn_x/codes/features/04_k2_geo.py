"""f2's hidden item key K2 (03_pairs: f2 ties form a fixed cross-category partition of items).
(a) Build K2 components from always-equal frequent-item pairs; size distribution; purity.
(b) Geography test: if K2 = location, the MSK-hour activity profile of feeds should shift by component (Russia spans 11 time zones).
(c) Item-embedding cosine of always-equal vs never-equal pairs, for f1 and f2 (same / different category).
(d) Stability of a user's dominant K2 component across his feeds.
Train uniform 400K-feed sample (seed 1)."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from pairlib import *
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

d, A, B, ukey, kinv, kcnt = build_pairs()
item = d["item"].astype(np.int64)
nk = len(ukey)
samecat = np.bincount(kinv, d["cat"][A] == d["cat"][B], minlength=nk) / kcnt
# representative rows per key (first occurrence)
first = np.full(nk, -1, np.int64); first[kinv[::-1]] = np.arange(len(kinv))[::-1]
ia = item[A[first]]; ib = item[B[first]]
ea = d["emb_idx"][A[first]]; eb = d["emb_idx"][B[first]]
E = np.load(ROOT / "data/competition/train/item_embeddings/data.memmap", mmap_mode="r") if False else None
meta = (ROOT / "data/competition/train/item_embeddings/data.meta").read_text()
import json; m = json.loads(meta)
E = np.memmap(ROOT / "data/competition/train/item_embeddings/data.memmap", dtype=m["dtype"], mode="r", shape=tuple(m["shape"]))

def cos_pairs(sel):
    a = ea[sel]; b = eb[sel]
    ok = (a >= 0) & (b >= 0)
    a = a[ok]; b = b[ok]
    if len(a) > 200_000:
        s = np.random.default_rng(0).choice(len(a), 200_000, replace=False); a = a[s]; b = b[s]
    Xa = np.asarray(E[np.sort(a)], dtype=np.float32); Xb = np.asarray(E[b], dtype=np.float32)
    # re-align a after sort
    oa = np.argsort(a); Xa2 = np.empty_like(Xa); Xa2[oa] = Xa
    Xa2 /= np.linalg.norm(Xa2, axis=1, keepdims=True) + 1e-9; Xb /= np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-9
    return float(np.mean(np.sum(Xa2 * Xb, 1))), len(a)

print("(c) item-embedding cosine of frequent pairs by equality class (>=3 valid co-occurrences)")
shares = {}
for k in ["f1", "f2", "f3"]:
    e_, n_ = pair_share(d, A, B, kinv, nk, k)
    ok = n_ >= 3
    sh = np.where(ok, e_ / np.maximum(n_, 1), np.nan)
    shares[k] = (sh, ok)
    for lab, cm in [("same cat", samecat == 1), ("diff cat", samecat == 0)]:
        out = []
        for cl, m in [("always", ok & (sh == 1)), ("never", ok & (sh == 0)), ("mixed", ok & (sh > 0) & (sh < 1))]:
            mm = m & cm
            if mm.sum() < 50: out.append(f"{cl}: n={mm.sum()}"); continue
            c, n = cos_pairs(mm)
            out.append(f"{cl}: cos {c:.3f} (n={mm.sum()})")
        print(f"  {k} {lab}: " + " | ".join(out))

# (a) K2 components from always-equal f2 pairs
sh, ok = shares["f2"]
alw = ok & (sh == 1)
nev = ok & (sh == 0)
uit, iinv = np.unique(np.r_[ia, ib], return_inverse=True)
na = len(ia)
ga = iinv[:na]; gb = iinv[na:]
G = coo_matrix((np.ones(alw.sum()), (ga[alw], gb[alw])), shape=(len(uit), len(uit)))
ncomp, lab = connected_components(G, directed=False)
cs = np.bincount(lab)
big = np.sort(cs)[::-1]
print(f"\n(a) K2 components over {len(uit)} frequent items: {ncomp} components; sizes top10 {big[:10].tolist()}; items in comps >=10: {cs[cs>=10].sum()} ({(cs>=10).sum()} comps)")
# purity: never-equal pairs inside the same component (violations)
inside = lab[ga] == lab[gb]
print(f"   never-equal pairs: {nev.sum()}, of which inside one component {np.sum(nev & inside)} ({np.mean(inside[nev]):.4f}); always-equal pairs crossing categories: {np.mean(samecat[alw]==0):.3f}")
# category composition of the largest components
cat_of = np.zeros(len(uit), np.int16)
cat_of[ga] = d["cat"][A[first]]; cat_of[gb] = d["cat"][B[first]]
for c in np.argsort(-cs)[:5]:
    cc = np.bincount(cat_of[lab == c], minlength=60)
    top = np.argsort(-cc)[:5]
    print(f"   comp size {cs[c]}: distinct categories {np.sum(cc>0)}; top cats " + ", ".join(f"{t}:{cc[t]/cs[c]:.2f}" for t in top))

# map item -> component for items in comps >= 10
comp_item = dict()
okc = cs[lab] >= 10
item2comp = pl.DataFrame({"item": uit[okc], "comp": lab[okc]})
# (b) per feed: dominant f2 level's component by majority vote of frequent items in the dominant level
df = pl.DataFrame({"feed": d["feed"], "item": d["item"].astype(np.int64), "f2": d["f2"], "ts": d["timestamp"], "user": d["user_id"]})
df = df.join(item2comp, on="item", how="left")
dom = (df.filter(pl.col("f2") > 0).group_by(["feed", "f2"]).agg(pl.len().alias("n"))
       .sort(["feed", "n"], descending=[False, True]).group_by("feed").first())
dd = df.join(dom.select(["feed", pl.col("f2").alias("f2dom")]), on="feed").filter((pl.col("f2") == pl.col("f2dom")) & pl.col("comp").is_not_null())
vote = dd.group_by(["feed", "comp"]).agg(pl.len().alias("n")).sort(["feed", "n"], descending=[False, True]).group_by("feed").agg(pl.col("comp").first(), pl.col("n").first(), pl.col("n").sum().alias("ntot"))
print(f"\n(b) feeds with a dominant-level component assigned: {len(vote)} of {d['n_feeds']}; vote purity (top comp share among voters) {np.mean(vote['n'].to_numpy()/vote['ntot'].to_numpy()):.3f}")
fm = df.group_by("feed").agg(pl.col("ts").first(), pl.col("user").first())
vote = vote.join(fm, on="feed")
vote = vote.with_columns((((pl.col("ts") + 3 * 3600) // 3600) % 24).alias("hour"))
tot = vote.group_by("hour").agg(pl.len().alias("n")).sort("hour")
glob = np.zeros(24); glob[tot["hour"].to_numpy()] = tot["n"].to_numpy(); glob /= glob.sum()
cc = vote.group_by("comp").agg(pl.len().alias("n")).filter(pl.col("n") >= 1500).sort("n", descending=True)
print(f"   components with >= 1500 dominant feeds: {len(cc)}")
rows = []
for c in cc["comp"].to_list():
    h = vote.filter(pl.col("comp") == c)["hour"].to_numpy()
    p = np.bincount(h, minlength=24) / len(h)
    # circular cross-correlation shift that best aligns p to the global profile
    best = max(range(-12, 12), key=lambda s: np.dot(np.roll(glob, s), p))
    night = p[[0, 1, 2, 3, 4, 5]].sum()  # MSK 00-05 share
    morning = p[[1, 2, 3, 4, 5, 6]].sum()
    rows.append((c, len(h), best, night, int(np.argmax(p))))
rows.sort(key=lambda r: r[2])
print("   comp | n feeds | best shift vs global MSK profile (h) | share of feeds 00-05 MSK | peak MSK hour")
for r in rows:
    print(f"   {r[0]} | {r[1]} | {r[2]:+d} | {r[3]:.3f} | {r[4]}")
print(f"   global MSK-hour profile (00..23): " + " ".join(f"{x:.3f}" for x in glob))
# (d) stability of dominant component within user
uu = vote.filter(pl.col("user") > 0).group_by("user").agg(pl.col("comp").n_unique().alias("nu"), pl.len().alias("nf")).filter(pl.col("nf") >= 2)
print(f"\n(d) users with >= 2 assigned feeds: {len(uu)}; share with a single dominant component across feeds: {np.mean(uu['nu'].to_numpy()==1):.3f}")
# baseline: random pairing of feeds -> probability two random feeds share a component
cvals = vote["comp"].to_numpy(); pc = np.bincount(cvals) / len(cvals)
print(f"   chance two random feeds share the dominant component: {np.sum(pc**2):.3f}")
np.save(OUT / "k2_item_comp.npy", np.c_[uit[okc], lab[okc]])
