"""CPU interim fix for the train-row availability leak: prew_* = pre_* restricted to history-warm candidates.

For every split (train/val/test) and each of the 9 pre_* features: NaN where it_hist_all_luo == 0 (the candidate never
appears in any other user's history); the within-feed z features are recomputed over the remaining candidates.
Same rule on all splits, so the 'content available or not' bit of train candidates (section 8b) can no longer leak.
Usage: python pretrain/make_prew.py
"""
import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/pretrain")
from diag_patch import zscore_nan

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; FEATS = ROOT / "data/feats"
NAMES = ["fe", "fe_z", "new", "new_z", "hmax32", "hmax8", "fe_noid", "id_contrib", "idmax32"]

for split in ["val", "test", "train"]:
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64)
    cold = np.load(FEATS / split / "it_hist_all_luo.npy") == 0
    raw = {}
    for n in NAMES:
        if n.endswith("_z"):
            continue
        x = np.load(FEATS / split / f"pre_{n}.npy").astype(np.float32)
        x[cold] = np.nan
        raw[n] = x
        np.save(FEATS / split / f"prew_{n}.npy", x)
    for zn, src in [("fe_z", "fe"), ("new_z", "new")]:
        np.save(FEATS / split / f"prew_{zn}.npy", zscore_nan(off, M, raw[src].copy()))
    print(split, "cold share", round(float(cold.mean()), 3), "prew_fe NaN share", round(float(np.isnan(raw["fe"]).mean()), 3), flush=True)
