"""hfill: shared helper. The fill of masked history vectors for one split (val / test), from out/fill_map_<split>.npz
(m0_count.py): history events with emb_idx < 0 whose item_id has a row in an EARLIER split's table (val: train; test: val,
then train). Returns the filled hist_emb_idx (int32) and the int8 rows to append to the split's own table:
  filled[ev] = n_table + j  where ext_table[j] = <source split table>[row]   (one ext row per distinct (source, row)).
Nothing here reads labels or any later split. Untouched events keep their original index (bit-identical arrays)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path("<repo>")
FLAT = ROOT / "data" / "flat"
COMP = ROOT / "data" / "competition"
HF = Path(__file__).resolve().parent
SRC_SPLIT = {0: "train", 1: "val"}


def item_table(split):
    meta = json.loads((COMP / split / "item_embeddings" / "data.meta").read_text())
    return np.memmap(COMP / split / "item_embeddings" / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def load_fill(split, n_table):
    """-> (hist_emb_idx filled int32, ext_table int8 [n_ext,128], info dict)."""
    assert split in ("val", "test")
    z = np.load(HF / "out" / f"fill_map_{split}.npz")
    ev, src, row = z["ev"], z["src"].astype(np.int64), z["row"].astype(np.int64)
    assert split == "test" or (src == 0).all(), "val is filled from the train table only"
    hemb = np.load(FLAT / split / "hist_emb_idx.npy")
    assert (hemb[ev] < 0).all()
    key = src * (1 << 40) + row
    ukey, inv = np.unique(key, return_inverse=True)
    usrc, urow = ukey >> 40, ukey & ((1 << 40) - 1)
    ext = np.empty((len(ukey), 128), np.int8)
    for s in np.unique(usrc):
        tab = item_table(SRC_SPLIT[int(s)])
        m = usrc == s
        r = urow[m]  # sorted within a source (ukey sorted)
        ext[m] = tab[r]
    filled = hemb.copy()
    filled[ev] = (n_table + inv).astype(np.int32)
    info = dict(split=split, n_table=int(n_table), n_ext=int(len(ukey)), n_events_filled=int(len(ev)),
                n_feeds_touched=int(len(z["feeds_rec"])))
    return filled, ext, info, z["feeds_rec"]
