"""Skeptic T5: contested contrasts on the existing holdout by train day (day 20713 follows the 2-day gap, item staleness ~4 d)."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path("<repo>")
VD = Path("/tmp/scratchpad/dpa/validation/out")
f = pl.read_parquet(ROOT / "data/flat/train/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); d = (f["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
p = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
p[M < 50] = 0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
hd = d[hold]
G3 = np.load(VD / "pfh_G3.npy")
for c in ["Q2a", "Q1a", "M2"]:
    x = np.load(VD / f"pfh_{c}.npy") - G3
    out = []
    for lab, m in [("<=20710", hd <= 20710), ("20713", hd == 20713), ("20714-15", hd >= 20714)]:
        out.append(f"{lab} {x[m].mean():+.5f}±{x[m].std(ddof=1)/np.sqrt(m.sum()):.5f} (n {m.sum()})")
    print(c, "-G3 holdout:", " | ".join(out))
