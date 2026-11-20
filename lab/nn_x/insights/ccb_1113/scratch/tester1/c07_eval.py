"""C07 nested LIN residual: [it_n7, it_trend, it_slope7, it_q95gap, exhaust] (z + pctrank + isnan) vs control [b].
First (information): FEATURES_REVIEW rule-5 check = test distribution under val-fit stats (test z-mean, share outside the val
1-99% range) for each column, plus the within-feed z version."""
from h import *
O = Path(HERE) / "out"
COLS = ["it_n7", "it_trend", "it_slope7", "it_q95gap", "exhaust"]
ft = pl.read_parquet(FLAT / "test/feeds.parquet"); Mt = ft["M"].to_numpy().astype(np.int64); ot = ft["cand_off"].to_numpy().astype(np.int64)
fv = pl.read_parquet(FLAT / "val/feeds.parquet"); Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy().astype(np.int64)
for c in COLS:
    v = np.load(O / f"c07_{c}_val.npy").astype(np.float64); t = np.load(O / f"c07_{c}_test.npy").astype(np.float64)
    mu, sd = np.nanmean(v), np.nanstd(v); lo, hi = np.nanpercentile(v, [1, 99])
    tz = (t - mu) / sd
    zt = feed_z_nanmin(t, ot, Mt); zv_ = feed_z_nanmin(v, ov, Mv)
    log(f"RULE5 {c:10s}: raw test z-mean under val stats {np.nanmean(tz):+.3f} (test finite {np.isfinite(t).mean():.3f} vs val {np.isfinite(v).mean():.3f});"
        f" test outside val 1-99% {np.mean((t[np.isfinite(t)] < lo) | (t[np.isfinite(t)] > hi)):.3f} | within-feed z: val sd {zv_.std():.3f} test sd {zt.std():.3f}")
H = Harness()
xs_v = [np.load(O / f"c07_{c}_val.npy").astype(np.float64) for c in COLS]
xs_h = [np.load(O / f"c07_{c}_hold.npy").astype(np.float64) for c in COLS]
for c, x, xh in zip(COLS, xs_v, xs_h):
    coverage(f"val {c}", x); coverage(f"hold {c}", xh)
H.run("C07_item_trend_lifecycle", xs_v, xs_h, out_jsonl=O / "results.jsonl")
