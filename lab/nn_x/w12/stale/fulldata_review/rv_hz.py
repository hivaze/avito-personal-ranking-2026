"""Reviewer (label-free): share of candidate rows whose popularity input changes under the hz view, holdout feeds vs other draw-set feeds."""
import numpy as np, polars as pl
R = "<repo>"
f = pl.read_parquet(f"{R}/data/flat/train/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
p = np.nan_to_num(np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0.0
cand = np.where((p > 0.9) & (M >= 50))[0]; hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
oth = np.sort(np.random.default_rng(5).choice(np.setdiff1d(np.where(p > 0.5)[0], hold), 30000, replace=False))
rows = lambda ids: np.concatenate([np.arange(o, o + m) for o, m in zip(off[ids], M[ids])])
lag = np.load(f"{R}/data/feats_new/train/hz_lag.npy", mmap_mode="r")
hz = np.load(f"{R}/data/feats_new/train/it_hist_hz_all.npy", mmap_mode="r"); luo = np.load(f"{R}/nn/feat_cache/train/it_hist_all_luo.npy", mmap_mode="r")
for nm, ids in (("holdout feeds", hold), ("30,000 other exploration-like train feeds", oth)):
    r = rows(ids); a, b = np.nan_to_num(np.asarray(hz[r], np.float64)), np.nan_to_num(np.asarray(luo[r], np.float64))
    lg = np.asarray(lag[ids]) if lag.shape[0] == len(M) else None
    print(f"{nm}: rows {len(r)}; it_hist_hz_all != it_hist_all_luo on {100 * (a != b).mean():.1f}% of rows; zero rate luo {np.mean(b == 0):.3f} -> hz {np.mean(a == 0):.3f}; "
          + (f"per-feed lag values {dict(zip(*np.unique(lg, return_counts=True)))}" if lg is not None else f"hz_lag shape {lag.shape}"))
