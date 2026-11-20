"""Label-free check for RH twins (--hzcd + --holdout_in_train): how do the horizon-limited popularity inputs
(data/feats_new/train/it_hist_hz_*.npy) of the 30,000 holdout feeds compare with the as-built LUO inputs, against other
exploration-like train feeds? CPU, memory-mapped, seconds.  Writes logs/hz_holdout.txt."""
import os, sys
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
sys.dont_write_bytecode = True
import numpy as np, polars as pl
R = "<repo>"; HERE = os.path.dirname(os.path.abspath(__file__))
f = pl.read_parquet(f"{R}/data/flat/train/feeds.parquet", columns=["M", "cand_off"])
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64); n = len(M)
p = np.nan_to_num(np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
other = np.sort(np.random.default_rng(1).choice(np.setdiff1d(cand, hold), 30000, replace=False))
rows = lambda fs: np.concatenate([np.arange(o, o + m) for o, m in zip(off[fs], M[fs])])
out = open(f"{HERE}/logs/hz_holdout.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
P("share of candidate rows whose hz value differs from the as-built LUO value | zero rate luo -> hz")
for nm, luo, hz in (("all", "it_hist_all_luo", "it_hist_hz_all"), ("et1", "it_hist_et1_luo", "it_hist_hz_et1")):
    a = np.load(f"{R}/nn/feat_cache/train/{luo}.npy", mmap_mode="r"); b = np.load(f"{R}/data/feats_new/train/{hz}.npy", mmap_mode="r")
    for tag, fs in (("30,000 holdout feeds", hold), ("30,000 other exploration-like train feeds (same rule)", other)):
        r = rows(fs); x = np.nan_to_num(np.asarray(a[r], np.float64)); y = np.nan_to_num(np.asarray(b[r], np.float64))
        P(f"  {nm:3s} {tag:55s}: changed {np.mean(x != y):.4f} | zero rate {np.mean(x == 0):.4f} -> {np.mean(y == 0):.4f} | mean log1p {np.log1p(np.maximum(x,0)).mean():.3f} -> {np.log1p(np.maximum(y,0)).mean():.3f} ({len(r)} rows)")
