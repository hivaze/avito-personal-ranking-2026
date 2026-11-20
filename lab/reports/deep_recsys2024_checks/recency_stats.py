"""Quick check of the 'recent data matters' lesson: item stats from the LAST days of train vs all 14 days, evaluated on val."""
import numpy as np
import polars as pl

T = "<repo>/data/flat/train"
V = "<repo>/data/flat/val"
ft = pl.read_parquet(f"{T}/feeds.parquet").select("fidx", "timestamp")
tts = ft["timestamp"].to_numpy()
ti = np.load(f"{T}/cand_item_id.npy", mmap_mode="r")
tf = np.load(f"{T}/cand_fidx.npy", mmap_mode="r")
ty = np.load(f"{T}/cand_target.npy", mmap_mode="r")
tr = pl.DataFrame({"item": np.asarray(ti), "t": tts[np.asarray(tf)], "y": np.asarray(ty).astype(np.int8)})
tmax = int(tr["t"].max())
print("train rows", tr.height, "days", (tmax - int(tr["t"].min())) / 86400)

fv = pl.read_parquet(f"{V}/feeds.parquet").select("fidx", "timestamp")
vi = np.load(f"{V}/cand_item_id.npy"); vf = np.load(f"{V}/cand_fidx.npy"); vy = np.load(f"{V}/cand_target.npy")
f1 = np.load(f"{V}/cand_f1.npy")
val = pl.DataFrame({"row": np.arange(len(vi)), "item": vi, "fidx": vf})

order_f = np.argsort(vf, kind="stable")
bounds = np.r_[0, np.flatnonzero(np.diff(vf[order_f])) + 1, len(vf)]


def ndcg(score):
    s = score[order_f]; t = vy[order_f]; res = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        tt = t[a:b]
        if tt.sum() == 0:
            continue
        o = np.lexsort((tt, -s[a:b])); g = tt[o]
        disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
        res.append((g * disc).sum() / disc[: int(tt.sum())].sum())
    return float(np.mean(res))


def frank(x):
    return pl.DataFrame({"f": vf, "x": x}).with_columns(
        (pl.col("x").rank("average").over("f") / pl.len().over("f")).alias("r"))["r"].to_numpy()


rf1 = frank(f1)
print(f"f1 alone {ndcg(f1):.4f}")
for days in (1, 3, 7, 14):
    sub = tr.filter(pl.col("t") > tmax - days * 86400)
    st = sub.group_by("item").agg(pl.len().alias("imp"), pl.col("y").sum().alias("pos"))
    m = val.join(st, on="item", how="left").sort("row").with_columns(
        pl.col("imp").fill_null(0), pl.col("pos").fill_null(0))
    imp = m["imp"].to_numpy().astype(np.float64); pos = m["pos"].to_numpy().astype(np.float64)
    prior = 0.024; a = 20.0
    ctr = (pos + a * prior) / (imp + a)
    cov = (imp > 0).mean()
    print(f"last {days:2d}d: cov={cov:.3f}  pos_cnt alone={ndcg(pos + 1e-6 * f1):.4f}  "
          f"rank(f1)+0.5rank(pos)={ndcg(rf1 + 0.5 * frank(pos)):.4f}  rank(f1)+0.5rank(ctr_sm)={ndcg(rf1 + 0.5 * frank(ctr)):.4f}")
