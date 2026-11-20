"""item_tab2: the pretraining content table + the TRAIN candidate embeddings (section 8c).
item_tab.npy (section 2) was built from all history events and the val/test candidate files only, so train candidates that
never appear in any history had no content (scored with the 'missing' vector) -- a label leak for train rows (8b).
Here every item that is a candidate in any split gets its own embedding. Items already present keep their row."""
import json, time
from pathlib import Path
import numpy as np, numba as nb
ROOT = Path("<repo>"); PT = ROOT / "data/pretrain"; FLAT = ROOT / "data/flat"; COMP = ROOT / "data/competition"
from build_corpus import first_src
t0 = time.time()
tab = np.load(PT / "item_tab.npy"); has = np.load(PT / "item_has.npy")
G = len(has)
meta = json.loads((COMP / "train/item_embeddings/data.meta").read_text())
T = np.memmap(COMP / "train/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
src = np.full(G, -1, np.int64); has_s = has.copy()
first_src(np.load(FLAT / "train/cand_item_id.npy"), np.load(FLAT / "train/cand_emb_idx.npy"), has_s, src)
new = np.where(has_s & ~has)[0]
order = np.argsort(src[new])
tab[new[order]] = T[src[new[order]]]
np.save(PT / "item_tab2.npy", tab); np.save(PT / "item_has2.npy", has_s)
print(f"added {len(new)} items from train candidates; has {has.sum()} -> {has_s.sum()}; {time.time()-t0:.0f}s", flush=True)
