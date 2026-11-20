"""V2: later-history leak - resolve the 108 vs 334 test-feed contradiction (user_id=0 contamination?),
and measure precision / NDCG effect on val with honest sources (val-only, mirrors what test can use)."""
import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

FLAT = Path("<repo>/data/flat")
BOUND = 1790497920  # public/private boundary (eda_shift_overlap)


def cands(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    fidx = np.load(FLAT / split / "cand_fidx.npy")
    return pl.DataFrame({"row": np.arange(len(fidx)), "fidx": fidx,
                         "user": f["user_id"].to_numpy().astype(np.int64)[fidx],
                         "tc": f["timestamp"].to_numpy().astype(np.int64)[fidx],
                         "item": np.load(FLAT / split / "cand_item_id.npy").astype(np.int64)}), f


def hist(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    hf = np.load(FLAT / split / "hist_fidx.npy")
    return pl.DataFrame({"user": f["user_id"].to_numpy().astype(np.int64)[hf],
                         "t": f["timestamp"].to_numpy().astype(np.int64)[hf],
                         "item": np.load(FLAT / split / "hist_item_id.npy").astype(np.int64),
                         "et": np.load(FLAT / split / "hist_event_type.npy").astype(np.int8)})


H = {s: hist(s) for s in ["val", "test"]}

# --- feed-level: who has a later feed in the same split
for split in ["val", "test"]:
    f = pl.read_parquet(FLAT / split / "feeds.parquet").select("fidx", "user_id", "timestamp")
    mx = f.group_by("user_id").agg(pl.col("timestamp").max().alias("tmax"))
    g = f.join(mx, on="user_id").with_columns((pl.col("timestamp") < pl.col("tmax")).alias("has_later"))
    nz = g.filter(pl.col("user_id") > 0)
    pub = pl.col("timestamp") < BOUND
    msg = f"{split}: feeds with a later same-split feed: user>0 {nz['has_later'].sum()}  (incl user0 {g['has_later'].sum()})"
    if split == "test":
        msg += f"; user>0 public {nz.filter(pub)['has_later'].sum()} / private {nz.filter(~pub)['has_later'].sum()}"
    print(msg, flush=True)

# --- candidate-level flags
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
for split, sources in [("val", ["val"]), ("val", ["val", "test"]), ("test", ["test"])]:
    c, f = cands(split)
    h = pl.concat([H[s] for s in sources])
    for incl0 in [False, True]:
        cc = c if incl0 else c.filter(pl.col("user") > 0)
        j = cc.join(h, on=["user", "item"], how="inner").filter(pl.col("t") > pl.col("tc"))
        rows = j["row"].unique().to_numpy()
        feeds = np.unique(c["fidx"].to_numpy()[rows])
        msg = f"{split} src={'+'.join(sources)} incl_user0={incl0}: flagged rows {len(rows)}, feeds {len(feeds)}"
        if split == "test":
            ts = f["timestamp"].to_numpy()[feeds]
            msg += f" (public {int((ts < BOUND).sum())}, private {int((ts >= BOUND).sum())})"
        else:
            msg += f", target rate {y[rows].mean():.3f}"
            if incl0:
                r0 = j.filter(pl.col("user") == 0)["row"].unique().to_numpy()
                msg += f"; user0-only rows {len(r0)} rate {y[r0].mean() if len(r0) else float('nan'):.3f}"
        print(msg, flush=True)
        if split == "val" and not incl0:
            fm = pl.read_parquet(FLAT / "val/feeds.parquet")
            M = fm["M"].to_numpy().astype(np.int64); off = fm["cand_off"].to_numpy()
            base = np.load(FLAT / "val/cand_f1.npy").astype(np.float64) + 1e-3 * np.load(FLAT / "val/cand_f5.npy")
            flag = np.zeros(len(y)); flag[rows] = 1
            b = ndcg_per_feed(base, y, off, M); a = ndcg_per_feed(base + 10 * flag, y, off, M)
            nn = np.load("<repo>/preds/nn_final_val_oof.npy").astype(np.float64)
            z = (nn - nn.mean()) / nn.std()
            bn = ndcg_per_feed(nn, y, off, M); an = ndcg_per_feed(z + 100 * flag, y, off, M)
            d = a - b; dn = an - bn
            print(f"   NDCG f1+1e-3f5: {b.mean():.5f} -> {a.mean():.5f} ({d.mean():+.5f}+-{d.std()/np.sqrt(len(d)):.5f});"
                  f"  nn_final: {bn.mean():.5f} -> {an.mean():.5f} ({dn.mean():+.5f}+-{dn.std()/np.sqrt(len(dn)):.5f});"
                  f" affected feeds {len(feeds)}: nn {bn[feeds].mean():.3f} -> {an[feeds].mean():.3f}", flush=True)
