"""Where FD16P20 loses NDCG on validation, and what the misses look like (lead, 2026-11-17).

Descriptive only (all 14,423 validation feeds, both halves); nothing here is fitted or selected.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kit  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
df = pl.concat([pl.read_parquet(HERE / "data/disc.parquet"), pl.read_parquet(HERE / "private/confirm.parquet")]).sort(["fidx", "row"])
off, sz = kit.feed_layout(df)
y = df["label"].to_numpy(); s = df["score"].to_numpy().astype(np.float64)
nd = kit.ndcg_per_feed(s, y, off, sz)
disc = 1.0 / np.log2(np.arange(2, 302))
npos = np.add.reduceat(y, off)
rand = np.array([p * disc[:m].mean() / disc[:p].sum() for p, m in zip(npos, sz)])
loss = 1 - nd
N = len(sz)
print(f"feeds {N}  NDCG {nd.mean():.5f}  random {rand.mean():.4f}  total loss {loss.sum():.0f} feed-units\n")

# 1. by feed size
print("1. BY FEED SIZE            feeds   NDCG  random  share of loss  share of gain over random")
gain = nd - rand
for lo, hi in [(5, 30), (31, 100), (101, 200), (201, 300)]:
    k = (sz >= lo) & (sz <= hi)
    print(f"   M {lo:3d}-{hi:3d}            {k.mean():6.1%}  {nd[k].mean():.3f}  {rand[k].mean():.3f}     {loss[k].sum() / loss.sum():6.1%}       {gain[k].sum() / gain.sum():6.1%}")

# 2. best positive of each feed, by class
d = df.with_columns(pl.int_range(pl.len()).alias("i"))
rank1_cat = d.filter(pl.col("rank") == 1).select("fidx", pl.col("cat").alias("top_cat"))
bp = (d.filter(pl.col("label") == 1).sort(["fidx", "rank"]).group_by("fidx", maintain_order=True).first()
      .join(rank1_cat, on="fidx"))
bp = bp.with_columns(
    (pl.col("cat") == pl.col("top_cat")).alias("in_top_cat"),
    pl.when(pl.col("cat_newest_idx") < 0).then(pl.lit("absent from history"))
      .when(pl.col("cat_newest_idx") < 5).then(pl.lit("in newest 5 events"))
      .when(pl.col("cat_newest_idx") < 32).then(pl.lit("in events 6-32"))
      .otherwise(pl.lit("only older than 32")).alias("recency"),
    (pl.col("h10").fill_nan(0.0) >= 0.95).alias("near_copy_recent"),
    (pl.col("xp_imp_c50") == 0).alias("cold"),
)
fpos = {f: k for k, f in enumerate(df["fidx"].to_numpy()[off])}
bp = bp.with_columns(pl.Series("loss", loss[[fpos[f] for f in bp["fidx"].to_list()]]),
                     pl.Series("ndcg", nd[[fpos[f] for f in bp["fidx"].to_list()]]))
print("\n2. THE FEED'S BEST POSITIVE, BY CLASS    share of feeds  median rank  feed NDCG  share of loss")
for col in ["in_top_cat", "recency", "near_copy_recent", "cold"]:
    g = bp.group_by(col).agg(pl.len().alias("n"), pl.col("rank").median().alias("med_rank"),
                             pl.col("ndcg").mean().alias("nd"), pl.col("loss").sum().alias("l")).sort(col)
    for r in g.iter_rows(named=True):
        print(f"   {col:17s} = {str(r[col]):20s} {r['n'] / N:6.1%}      {r['med_rank']:6.0f}     {r['nd']:.3f}     {r['l'] / loss.sum():6.1%}")

# 3. oracles on this model's order
cat = df["cat"].to_numpy(); rk = df["rank"].to_numpy()
cat_or, within_or = np.empty(N), np.empty(N)
for k, (o, m) in enumerate(zip(off, sz)):
    yy, cc, rr = y[o:o + m], cat[o:o + m], rk[o:o + m]
    order = np.argsort(rr)                                  # model order
    pos_cats = set(cc[yy == 1].tolist())
    first = np.array([c in pos_cats for c in cc[order]])
    o1 = order[np.argsort(~first, kind="stable")]           # categories with a positive first, model order inside
    t1 = yy[o1]; cat_or[k] = (t1 * disc[:m]).sum() / disc[:int(yy.sum())].sum()
    o2 = order.copy()                                       # keep each category's slots, best items of it first
    for c in np.unique(cc):
        slots = np.nonzero(cc[order] == c)[0]
        items = order[slots]
        o2[slots] = items[np.argsort(-yy[items], kind="stable")]
    t2 = yy[o2]; within_or[k] = (t2 * disc[:m]).sum() / disc[:int(yy.sum())].sum()
print(f"\n3. ORACLES ON THE MODEL'S ORDER: model {nd.mean():.4f} | categories with a positive first {cat_or.mean():.4f} "
      f"| model's category slots, perfect order inside {within_or.mean():.4f} | perfect 1.0")

# 4. the misses that matter: best positive at rank 31+ in feeds of 100+
big = bp.filter((pl.col("M") >= 100))
miss = big.filter(pl.col("rank") >= 31)
hit = big.filter(pl.col("rank") <= 3)
print(f"\n4. FEEDS OF 100+ CANDIDATES: {big.height} feeds; best positive at rank 31+ in {miss.height / big.height:.0%}, at 1-3 in {hit.height / big.height:.0%}")
print(f"   misses: in the rank-1 item's category {miss['in_top_cat'].mean():.0%}; median category size {miss['cat_n_in_feed'].median():.0f}, "
      f"median rank inside its category {miss['cat_rank_in_feed'].median():.0f}")
neg1 = d.filter((pl.col("rank") == 1) & (pl.col("label") == 0) & (pl.col("M") >= 100))
cols = ["h1", "h5", "hmean5", "ucos", "f1", "f3", "f4", "f5", "xp_imp_c50", "cat_hist_share"]
print("   medians:                 " + " ".join(f"{c:>9s}" for c in cols))
for name, t in [("positives at rank 1-3", hit), ("rank-1 items, negative", neg1), ("positives at rank 31+", miss)]:
    print(f"   {name:24s} " + " ".join(f"{t[c].fill_nan(None).median():9.3f}" for c in cols))
allc = d.filter(pl.col("M") >= 100)
print(f"   {'all candidates':24s} " + " ".join(f"{allc[c].fill_nan(None).median():9.3f}" for c in cols))

# 5. calibration of the head
print("\n5. POSITIVE RATE BY RANK vs A WITHIN-FEED SOFTMAX OF THE SCORE (slope 1, no fitting)")
e = np.exp(s - np.maximum.reduceat(s, off).repeat(sz))
p = e / np.add.reduceat(e, off).repeat(sz) * npos.repeat(sz)
for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 30), (31, 100), (101, 300)]:
    k = (rk >= lo) & (rk <= hi)
    print(f"   ranks {lo:3d}-{hi:3d}: actual {y[k].mean():.4f}  softmax {p[k].mean():.4f}  ratio {y[k].mean() / p[k].mean():.2f}")
