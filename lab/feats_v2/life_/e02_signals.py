"""Exploration step 2: simple lifecycle signals for val/test candidates and their val single-feature NDCG."""
import sys, time
sys.path.insert(0, "<repo>/src")
from common import *

SCR = Path("/tmp/scratchpad")
t0 = time.time()
A = pl.read_parquet(SCR / "app_valtest_items.parquet")
TR_END = 1789862400  # Sun Sep 20 00:00 MSK (= Sep 19 21:00 UTC)
rng = np.random.default_rng(0)

# static train aggregates per item
tr = A.filter(pl.col("split") == 0).with_columns(((pl.col("t") + 3 * 3600) // 86400).alias("day"))
aggc = tr.filter(pl.col("src") == 0).group_by("item").agg(
    pl.col("t").min().alias("trc_first"), pl.col("t").max().alias("trc_last"), pl.len().alias("trc_n"),
    pl.col("day").n_unique().alias("trc_ndays"), (pl.col("t") >= TR_END - 86400).sum().alias("trc_nlast1d"))
aggh = tr.filter(pl.col("src") == 1).group_by("item").agg(
    pl.col("t").min().alias("trh_first"), pl.col("t").max().alias("trh_last"), pl.len().alias("trh_n"),
    pl.col("user").n_unique().alias("trh_nusers"),
    pl.col("t").filter(pl.col("hpos") < 8).max().alias("trh_last_fresh"),
    pl.col("t").filter(pl.col("hpos") < 8).min().alias("trh_first_fresh"),
    pl.col("hpos").mean().alias("trh_mean_hpos"))
print("train aggs", f"{time.time()-t0:.0f}s", flush=True)

sp = A.filter(pl.col("split") > 0)


def rows(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    fi = np.load(FLAT / split / "cand_fidx.npy")
    return pl.DataFrame({"row": np.arange(len(fi)), "fidx": fi, "item": np.load(FLAT / split / "cand_item_id.npy"),
                         "t": f["timestamp"].to_numpy()[fi], "user": f["user_id"].to_numpy()[fi],
                         "split": np.full(len(fi), 1 if split == "val" else 2, np.int8)})


def build(split):
    R = rows(split)
    j = R.join(sp, on="item", suffix="_o")
    j = j.filter(~((pl.col("split_o") == pl.col("split")) & (pl.col("fidx_o") == pl.col("fidx"))))
    j = j.filter(~((pl.col("user_o") == pl.col("user")) & (pl.col("user") != 0)))
    j = j.with_columns((pl.col("t_o") - pl.col("t")).alias("dt"), (pl.col("src") % 2 == 1).alias("ish"))
    g = j.group_by("row").agg(
        ((pl.col("dt") > 0) & ~pl.col("ish")).sum().alias("sp_c_fut"),
        ((pl.col("dt") < 0) & ~pl.col("ish")).sum().alias("sp_c_past"),
        ((pl.col("dt") > 0) & pl.col("ish")).sum().alias("sp_h_fut"),
        ((pl.col("dt") <= 0) & pl.col("ish")).sum().alias("sp_h_past"),
        ((pl.col("dt") > 0) & (pl.col("dt") <= 86400) & ~pl.col("ish")).sum().alias("sp_c_fut24"),
        ((pl.col("dt") > 0) & (pl.col("dt") <= 86400) & pl.col("ish")).sum().alias("sp_h_fut24"),
        ((pl.col("dt") > 0) & (pl.col("dt") <= 86400) & pl.col("ish") & (pl.col("et") == 1)).sum().alias("sp_h1_fut24"),
        pl.col("dt").max().alias("sp_last_dt"),
        pl.col("dt").min().alias("sp_first_dt"),
        pl.col("dt").filter(pl.col("dt") < 0).max().alias("sp_prev_dt"),
        pl.col("dt").filter(pl.col("dt") > 0).min().alias("sp_next_dt"),
        pl.col("hpos").filter(pl.col("ish") & (pl.col("dt") > 0)).min().alias("sp_h_fut_minhpos"),
    )
    R = R.join(g, on="row", how="left").join(aggc, on="item", how="left").join(aggh, on="item", how="left").sort("row")
    R = R.with_columns(
        ((pl.col("t") - pl.min_horizontal("trc_first", "trh_first")) / 3600).alias("tr_age_h"),
        ((TR_END - pl.max_horizontal("trc_last", "trh_last")) / 3600).alias("tr_lastgap_h"),
        ((TR_END - pl.col("trc_last")) / 3600).alias("trc_lastgap_h"),
        ((TR_END - pl.col("trh_last_fresh")) / 3600).alias("trh_lastfresh_gap_h"),
        ((pl.col("trc_last") - pl.col("trc_first")) / 3600).alias("trc_span_h"),
        ((TR_END - pl.col("trh_first_fresh")) / 3600).alias("trh_firstfresh_age_h"),
    )
    return R


def sf_ndcg(x, y, off, M, seed=0):
    x = np.asarray(x, np.float64)
    noise = np.random.default_rng(seed).random(len(x)) * 1e-9
    res = []
    for sgn in (1, -1):
        v = np.where(np.isnan(x), -np.inf, sgn * x)
        # NaN -> bottom, random tie-break
        r = np.argsort(np.argsort(v + 0)) if False else None
        s = np.where(np.isinf(v), -1e30, v)
        res.append(ndcg(s + noise * (np.nanstd(x) + 1e-9) * 0 + noise, y, off, M))
    return res


V = build("val")
T = build("test")
print("built", f"{time.time()-t0:.0f}s", flush=True)
V.write_parquet(SCR / "e02_val.parquet"); T.write_parquet(SCR / "e02_test.parquet")
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
print("random", sf_ndcg(np.zeros(len(y)), y, off, M))
feeds_test = pl.read_parquet(FLAT / "test/feeds.parquet")
for c in V.columns:
    if c in ("row", "fidx", "item", "t", "user", "split"):
        continue
    x = V[c].cast(pl.Float64).to_numpy()
    xt = T[c].cast(pl.Float64).to_numpy()
    nn = np.isfinite(x)
    a, b = sf_ndcg(x, y, off, M)
    # pos rate by nonnull and quartile
    q = np.nanquantile(x, [0.25, 0.5, 0.75]) if nn.any() else [np.nan] * 3
    print(f"{c:22s} nd+={a:.4f} nd-={b:.4f} cov_val={nn.mean():.3f} cov_test={np.isfinite(xt).mean():.3f} "
          f"pos|null={y[~nn].mean() if (~nn).any() else np.nan:.4f} pos|nn={y[nn].mean() if nn.any() else np.nan:.4f} "
          f"q_val={np.round(q,1)} q_test={np.round(np.nanquantile(xt,[0.25,0.5,0.75]),1) if np.isfinite(xt).any() else None}",
          flush=True)
