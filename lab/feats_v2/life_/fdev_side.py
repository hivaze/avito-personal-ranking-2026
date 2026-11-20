"""Side experiment (not lifecycle): item-centered f-score deviations. For val/test rows: f_k(now) minus the mean of
f_k over the item's train candidate appearances (k = 1, 2, 3, 5), the train means themselves, and f5 drift
(now - last train value; first->last train slope). Written as life_x_*; label-free, uses only train (strictly earlier).
Only val/test were built (screened, then dropped)."""
import numpy as np, polars as pl
d = "<repo>/data/flat"
F = "<repo>/data/feats"
f = pl.read_parquet(f"{d}/train/feeds.parquet"); ts = f["timestamp"].to_numpy()
fi = np.load(f"{d}/train/cand_fidx.npy")
cols = {"item": np.load(f"{d}/train/cand_item_id.npy"), "t": ts[fi]}
for k in (1, 2, 3, 5):
    cols[f"f{k}"] = np.load(f"{d}/train/cand_f{k}.npy")
df = pl.DataFrame(cols); del fi
g = df.group_by("item").agg(pl.col("t").first().alias("t1"), pl.col("t").last().alias("t2"),
     *[pl.col(f"f{k}").first().alias(f"a{k}") for k in (5,)], *[pl.col(f"f{k}").last().alias(f"b{k}") for k in (5,)],
     *[pl.col(f"f{k}").mean().alias(f"m{k}") for k in (1, 2, 3, 5)])
del df
for s in ["val", "test"]:
    it = np.load(f"{d}/{s}/cand_item_id.npy")
    q = pl.DataFrame({"item": it, "row": np.arange(len(it)), **{f"f{k}": np.load(f"{d}/{s}/cand_f{k}.npy") for k in (1, 2, 3, 5)}})
    q = q.join(g, on="item", how="left").sort("row")
    out = {
        "x_d5_last": q["f5"] - q["b5"],
        "x_d5_mean": q["f5"] - q["m5"],
        "x_slope5": (q["b5"] - q["a5"]) / ((q["t2"] - q["t1"]) / 86400 + 1),
        "x_d1_mean": q["f1"] - q["m1"], "x_d2_mean": q["f2"] - q["m2"], "x_d3_mean": q["f3"] - q["m3"],
        "x_m1": q["m1"], "x_m2": q["m2"], "x_m3": q["m3"],
    }
    for n, v in out.items():
        np.save(f"{F}/{s}/life_{n}.npy", v.cast(pl.Float32).fill_null(np.nan).to_numpy())
print("done")
