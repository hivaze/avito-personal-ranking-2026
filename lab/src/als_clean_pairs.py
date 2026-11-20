"""Leak-free ALS input: drop (user, item) pairs where item is a candidate of one of the user's val/test feeds.

Such pairs can only come from another snapshot of the same user (candidates are filtered from their own feed's history);
pairs from LATER snapshots carry the feed's outcome, which does not transfer to test. user 0 (pooled) is treated the same.
"""
import numpy as np, polars as pl
FLAT = "<repo>/data/flat"
OUT = "<repo>/eda/collab_stats/out"
g = pl.read_parquet(f"{OUT}/ui_interactions.parquet")
c = []
for s in ["val", "test"]:
    f = pl.read_parquet(f"{FLAT}/{s}/feeds.parquet")
    u = f["user_id"].to_numpy()[np.load(f"{FLAT}/{s}/cand_fidx.npy")].astype(np.int32)
    c.append(pl.DataFrame({"u": u, "i": np.load(f"{FLAT}/{s}/cand_item_id.npy")}))
c = pl.concat(c).unique()
clean = g.join(c, on=["u", "i"], how="anti")
print("pairs before", len(g), "after", len(clean), "removed", len(g) - len(clean))
clean.write_parquet(f"{OUT}/ui_interactions_clean.parquet")
