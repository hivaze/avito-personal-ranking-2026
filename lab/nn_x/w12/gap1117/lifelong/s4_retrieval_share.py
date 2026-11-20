"""Step 3 (label-free, CPU): how the top-16 retrieval (HistRetrieval: top-m history events by raw cosine to the candidate,
events without a vector excluded, no de-duplication in the final students) would treat extension events, and how new they are.
For every extended feed (capped-only k0 arrays) of val and test (and 3,000 train feeds): share of the 16 retrieval slots taken by
extension events, share of candidates with >= 1 extension event in their top 16, share of extension events whose item already
occurs among the stored 256 events of the same feed. No label is read.
  python s4_retrieval_share.py > out_s4_retrieval_share.txt
"""
import json
import os
import sys

sys.dont_write_bytecode = True
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_k] = "12"
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

DX = "<repo>/artifacts/data/feats_new/histext_1117"
FLAT = "<repo>/data/flat"
COMP = "<repo>/data/competition"
rng = np.random.default_rng(17)


def table(split):
    meta = json.loads(open(f"{COMP}/{split}/item_embeddings/data.meta").read())
    return np.memmap(f"{COMP}/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def unit(x):
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-3)


for split in ("val", "test", "train"):
    f = pl.read_parquet(f"{FLAT}/{split}/feeds.parquet")
    M, coff, H, hoff = (f[c].to_numpy().astype(np.int64) for c in ("M", "cand_off", "H", "hist_off"))
    n = np.load(f"{DX}/{split}/ext_n.npy").astype(np.int64)
    n = np.where(H == 256, n, 0)
    xoff = np.load(f"{DX}/{split}/ext_off.npy")
    xemb = np.load(f"{DX}/{split}/ext_emb_idx.npy", mmap_mode="r")
    xit = np.load(f"{DX}/{split}/ext_item_id.npy", mmap_mode="r")
    xtab = np.load(f"{DX}/{split}/ext_table.npy")
    tab = table(split)
    nt = tab.shape[0]
    cemb = np.load(f"{FLAT}/{split}/cand_emb_idx.npy", mmap_mode="r")
    hemb = np.load(f"{FLAT}/{split}/hist_emb_idx.npy", mmap_mode="r")
    hit = np.load(f"{FLAT}/{split}/hist_item_id.npy", mmap_mode="r")
    feeds = np.where(n > 0)[0]
    if split == "train":
        feeds = rng.choice(feeds, 3000, replace=False)

    def vec(idx):
        out = np.zeros((len(idx), 128), np.float32)
        ok = idx >= 0
        own = ok & (idx < nt)
        if own.any():
            out[own] = tab[np.asarray(idx[own])]
        ex = ok & (idx >= nt)
        if ex.any():
            out[ex] = xtab[np.asarray(idx[ex]) - nt]
        return unit(out), ok

    slots_ext, slots_all, cand_with, cand_all, dup, xev, new_cat = 0, 0, 0, 0, 0, 0, 0
    for fi in feeds:
        ce, cok = vec(np.asarray(cemb[coff[fi]:coff[fi] + M[fi]]).astype(np.int64))
        he, hok = vec(np.asarray(hemb[hoff[fi]:hoff[fi] + 256]).astype(np.int64))
        xe, xok = vec(np.asarray(xemb[xoff[fi]:xoff[fi] + n[fi]]).astype(np.int64))
        E = np.concatenate([he, xe])
        ok = np.concatenate([hok, xok])
        S = ce @ E.T
        S[:, ~ok] = -2.0
        top = np.argpartition(-S, 15, axis=1)[:, :16]
        val = np.take_along_axis(S, top, 1) > -1.5
        isx = (top >= 256) & val
        m = cok
        slots_ext += int(isx[m].sum())
        slots_all += int(val[m].sum())
        cand_with += int(isx[m].any(1).sum())
        cand_all += int(m.sum())
        st_items = set(np.asarray(hit[hoff[fi]:hoff[fi] + 256]).tolist())
        xi = np.asarray(xit[xoff[fi]:xoff[fi] + n[fi]])
        dup += int(np.isin(xi, list(st_items)).sum())
        xev += len(xi)
    print(f"{split}: extended capped feeds read {len(feeds)}; extension events {xev}; of them on an item already among the stored 256 events "
          f"{dup / xev:.3f}; retrieval (top-16 over stored + extension, as patched): slots taken by extension events {slots_ext / slots_all:.4f} "
          f"(extension = {np.mean(n[feeds]) / (256 + np.mean(n[feeds])):.3f} of the tokens); candidates with >= 1 extension event in their top 16: "
          f"{cand_with / cand_all:.3f}", flush=True)
