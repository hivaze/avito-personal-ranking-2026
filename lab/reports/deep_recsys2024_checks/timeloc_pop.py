"""Quick check: time-localized transductive item popularity from OTHER feeds' recent history events
(EB-NeRD :D 'global_article_-5m_5m' analog). Val only (labels available)."""
import numpy as np
import polars as pl

D = "<repo>/data/flat/val"
F = "<repo>/data/feats/val"
feeds = pl.read_parquet(f"{D}/feeds.parquet").select("fidx", "timestamp", "user_id")
ts = feeds["timestamp"].to_numpy()
uid = feeds["user_id"].to_numpy()

c_item = np.load(f"{D}/cand_item_id.npy")
c_fidx = np.load(f"{D}/cand_fidx.npy")
y = np.load(f"{D}/cand_target.npy")
f1 = np.load(f"{D}/cand_f1.npy")
h_item = np.load(f"{D}/hist_item_id.npy")
h_fidx = np.load(f"{D}/hist_fidx.npy")
h_pos = np.load(f"{D}/hist_hpos.npy")
h_et = np.load(f"{D}/hist_event_type.npy")
print("hpos range", h_pos.min(), h_pos.max(), "events", len(h_item))

cand = pl.DataFrame({"row": np.arange(len(c_item)), "item": c_item, "fidx": c_fidx,
                     "t": ts[c_fidx], "u": uid[c_fidx]})
ev = pl.DataFrame({"item": h_item, "efidx": h_fidx, "hpos": h_pos, "et": h_et,
                   "te": ts[h_fidx], "eu": uid[h_fidx]})
ev = ev.filter(pl.col("item").is_in(cand["item"].unique().implode()))
j = cand.join(ev, on="item", how="inner").filter(pl.col("eu") != pl.col("u"))
j = j.with_columns((pl.col("te") - pl.col("t")).alias("dt"))
print("joined pairs", j.height)

feats = {}
for K in (8, 32, 256):
    for W in (3600, 6 * 3600, 86400, 10**9):
        for side in ("past", "fut", "both"):
            cond = (pl.col("hpos") < K) & (pl.col("dt").abs() <= W)
            if side == "past":
                cond = cond & (pl.col("dt") <= 0)
            elif side == "fut":
                cond = cond & (pl.col("dt") > 0)
            g = j.filter(cond).group_by("row").len()
            a = np.zeros(len(c_item), dtype=np.float32)
            a[g["row"].to_numpy()] = g["len"].to_numpy()
            feats[f"K{K}_W{W if W < 10**9 else 'all'}_{side}"] = a

# NDCG (binary gain, full list, pessimistic ties) per feed
order_f = np.argsort(c_fidx, kind="stable")
assert (np.diff(c_fidx[order_f]) >= 0).all()
bounds = np.r_[0, np.flatnonzero(np.diff(c_fidx[order_f])) + 1, len(c_fidx)]


def ndcg(score):
    s = score[order_f]; t = y[order_f]
    res = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        tt = t[a:b]
        if tt.sum() == 0:
            continue
        ss = s[a:b]
        # pessimistic ties: negatives first among equal scores
        o = np.lexsort((tt, -ss))
        g = tt[o]
        disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
        dcg = (g * disc).sum()
        idcg = disc[: int(tt.sum())].sum()
        res.append(dcg / idcg)
    return float(np.mean(res))


def frank(x):
    # within-feed rank normalized 0..1 (average ties)
    out = np.empty(len(x), dtype=np.float64)
    df = pl.DataFrame({"f": c_fidx, "x": x}).with_columns(
        (pl.col("x").rank("average").over("f") / pl.len().over("f")).alias("r"))
    return df["r"].to_numpy()


base = ndcg(f1)
rf1 = frank(f1)
print(f"f1 alone NDCG={base:.4f}")
hall = np.load(f"{F}/it_hist_all.npy") if __import__('os').path.exists(f"{F}/it_hist_all.npy") else None
if hall is not None:
    print(f"it_hist_all: alone={ndcg(hall + 1e-6 * f1):.4f}  rank(f1)+0.5rank={ndcg(rf1 + 0.5 * frank(hall)):.4f}")
for k, a in feats.items():
    nz = a > 0
    print(f"{k:22s} cov={nz.mean():.4f} tr_nz={y[nz].mean() if nz.any() else 0:.4f} tr_z={y[~nz].mean():.4f} "
          f"alone={ndcg(a + 1e-6 * f1):.4f} rank(f1)+0.5rank={ndcg(rf1 + 0.5 * frank(a)):.4f}")
