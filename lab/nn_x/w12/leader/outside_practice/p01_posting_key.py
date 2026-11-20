"""p01: is (leaf category, posting time) a near-unique key for test candidate listings?
Necessary condition for matching the anonymised listings to a public catalogue page (which shows a publication
date/time and a view counter). Read-only, CPU only. Posting time = feed timestamp - age(f4), age via the 64-knot
map of lab/nn_x/audit/external (life_f4_age_knots.npy). Counts are over the DATASET's own test items only, so
uniqueness here is an upper bound on uniqueness in the real catalogue."""
import numpy as np, polars as pl, time
from pathlib import Path

ROOT = Path("<repo>/artifacts/data")
FLAT = ROOT / "flat" / "test"
knots = np.load(ROOT / "feats" / "life_f4_age_knots.npy")
K = 64; CAP = 63 / 64 - 1e-6
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def age_hours(x):
    x = np.asarray(x, dtype=np.float64)
    u = x * K
    k = np.clip(np.floor(u).astype(int), 1, K - 2)
    w = u - k
    a = knots[k] * (1 - w) + knots[k + 1] * w
    return np.where(x < CAP, a, np.nan)


f = pl.read_parquet(FLAT / "feeds.parquet")
ts = f["timestamp"].to_numpy().astype(np.int64)
fidx = np.load(FLAT / "cand_fidx.npy").astype(np.int64)
item = np.load(FLAT / "cand_item_id.npy").astype(np.int64)
cat = np.load(FLAT / "cand_category_id.npy").astype(np.int64)
age = age_hours(np.load(FLAT / "cand_f4.npy"))
post = ts[fidx] - age * 3600.0
ok = np.isfinite(post)
log(f"test candidate rows {len(item):,}; rows with finite age {ok.mean():.4f}; distinct items {len(np.unique(item)):,}")
for d in (1, 7, 30, 365):
    log(f"  rows with listing age > {d:3d} d at feed time: {(np.nan_to_num(age, nan=1e9) > 24 * d).mean():.3f}")

# per-item consistency of the posting time across appearances
df = pl.DataFrame({"item": item[ok], "cat": cat[ok], "post": post[ok]})
g = df.group_by("item").agg(pl.col("post").median().alias("p"), (pl.col("post").max() - pl.col("post").min()).alias("rng"),
                             pl.len().alias("n"), pl.col("cat").first().alias("cat"))
multi = g.filter(pl.col("n") > 1)
log(f"items seen >1 time: {multi.height:,}; posting-time range across appearances: median {multi['rng'].median():.2f} s, "
    f"p90 {multi['rng'].quantile(0.9):.2f} s, p99 {multi['rng'].quantile(0.99):.2f} s")

# uniqueness of (category, posting time bucket) among distinct test items
for res in (1, 60, 600):
    b = g.with_columns((pl.col("p") / res).floor().cast(pl.Int64).alias("b"))
    c = b.group_by(["cat", "b"]).agg(pl.len().alias("k"))
    bb = b.join(c, on=["cat", "b"])
    k = bb["k"].to_numpy()
    log(f"key (category, posting time // {res:4d} s): items alone in their cell {np.mean(k == 1):.3f}; mean cell size {k.mean():.2f}; "
        f"share in cells of <= 3 {np.mean(k <= 3):.3f}")
