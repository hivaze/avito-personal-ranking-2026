"""f1 vs f3 nesting on val: floor implications, constancy of f1 inside (feed, category) by f3 level (backoff happens at low
affinity?), number of distinct f1 levels per (feed, category) vs group size, and f1 tie-groups vs item-embedding similarity."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *
import json

d = load("val")
f1 = d["f1"]; f3 = d["f3"]
fl1 = np.abs(f1 - 1/56) < 1e-6; fl3 = np.abs(f3 - 1/62) < 1e-6
print(f"P(f1 floor | f3 floor) {fl1[fl3].mean():.3f}; P(f3 floor | f1 floor) {fl3[fl1].mean():.3f}; P(f1 floor) {fl1.mean():.3f}; P(f3 floor) {fl3.mean():.3f}")
key = d["feed"].astype(np.int64) * 64 + d["cat"]
df = pl.DataFrame({"key": key, "f1": f1, "f3": f3, "fl1": fl1, "emb": d["emb_idx"], "y": d["target"]})
g = df.group_by("key").agg(pl.len().alias("n"), pl.col("f1").n_unique().alias("nu"), pl.col("f3").first(), pl.col("fl1").mean().alias("flshare"),
                           pl.col("f1").mean().alias("f1m"), pl.col("f1").std().alias("f1sd"))
g = g.filter(pl.col("n") >= 3)
q = np.quantile(g["f3"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
b = np.searchsorted(q, g["f3"].to_numpy(), side="right")
n = g["n"].to_numpy(); nu = g["nu"].to_numpy(); fls = g["flshare"].to_numpy()
print("(feed, category) groups with >= 3 rows, by f3 quintile: share fully constant f1 | distinct/n | f1 floor share | mean f1 | mean f1 sd")
for i in range(5):
    m = b == i
    print(f"   q{i}: groups {m.sum()}, const {np.mean(nu[m]==1):.3f}, distinct/n {np.mean(nu[m]/n[m]):.3f}, f1 floor {np.mean(fls[m]):.3f}, mean f1 {g['f1m'].to_numpy()[m].mean():.3f}, f1 sd {np.nanmean(g['f1sd'].to_numpy()[m]):.4f}")
# within (feed,cat): is the target higher for items whose f1 is above the group's median? (finer preference signal)
df = df.with_columns((pl.col("f1") - pl.col("f1").median().over("key")).alias("dev"), pl.len().over("key").alias("n"), pl.col("f1").n_unique().over("key").alias("nu"))
sub = df.filter((pl.col("n") >= 5) & (pl.col("nu") >= 3))
dev = sub["dev"].to_numpy(); y = sub["y"].to_numpy()
print(f"within (feed,cat) groups with >= 5 rows and >= 3 f1 levels: rows {len(sub)}; target rate f1 above group median {y[dev>0].mean()*100:.2f}%, at median {y[dev==0].mean()*100:.2f}%, below {y[dev<0].mean()*100:.2f}%")
# f1 tie groups vs embedding cosine inside (feed, cat)
meta = json.loads((ROOT / "data/competition/val/item_embeddings/data.meta").read_text())
E = np.memmap(ROOT / "data/competition/val/item_embeddings/data.memmap", dtype=meta["dtype"], mode="r", shape=tuple(meta["shape"]))
rng = np.random.default_rng(0)
o = np.lexsort((rng.random(len(key)), key)); ks = key[o]
same_next = ks[1:] == ks[:-1]
ia = o[:-1][same_next]; ib = o[1:][same_next]
ok = (d["emb_idx"][ia] >= 0) & (d["emb_idx"][ib] >= 0) & ~fl1[ia] & ~fl1[ib]
ia = ia[ok]; ib = ib[ok]
Xa = np.asarray(E[d["emb_idx"][ia]], np.float32); Xb = np.asarray(E[d["emb_idx"][ib]], np.float32)
c = np.sum(Xa * Xb, 1) / (np.linalg.norm(Xa, axis=1) * np.linalg.norm(Xb, axis=1) + 1e-9)
eq = f1[ia] == f1[ib]
print(f"random same-(feed,cat) non-floor pairs: n {len(c)}; f1 equal share {eq.mean():.3f}; mean cos equal {c[eq].mean():.3f} vs unequal {c[~eq].mean():.3f}")
dd = np.abs(f1[ia] - f1[ib])
for lo, hi in [(0, 1e-9), (1e-9, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 1)]:
    m = (dd >= lo) & (dd < hi)
    print(f"   |f1 diff| in [{lo},{hi}): n {m.sum()}, mean cos {c[m].mean():.3f}")
