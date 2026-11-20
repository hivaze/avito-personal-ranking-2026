import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg, ndcg_per_feed

FLAT = Path("<repo>/data/flat")
FEATS = Path("<repo>/data/feats")


def feed_meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    return f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy()


def load_feats(split, names, rows=None):
    cols = []
    for n in names:
        x = np.load(FEATS / split / f"{n}.npy", mmap_mode="r")
        cols.append(np.asarray(x if rows is None else x[rows], dtype=np.float32))
    return np.stack(cols, axis=1)


def all_feature_names(split="val"):
    return sorted(p.stem for p in (FEATS / split).glob("*.npy"))


def swap_luo(names):
    """Replace leaky global history counts (it_hist_*) by their leave-user-later-out versions (src/features_luo.py)."""
    out = []
    for n in names:
        if n.startswith("it_hist_all"):
            out.append(n.replace("it_hist_all", "it_hist_all_luo"))
        elif n.startswith("it_hist_et") and not n.endswith("_luo"):
            out.append(n + "_luo")
        else:
            out.append(n)
    return out
