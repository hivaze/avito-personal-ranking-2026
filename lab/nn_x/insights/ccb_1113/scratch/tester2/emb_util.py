"""Own-history side loader (B10): history + candidate arrays and L2-normalised float32 item vectors for a feed subset."""
import json
import numpy as np, polars as pl
from pathlib import Path

FLAT = Path("<repo>/data/flat")
COMP = Path("<repo>/data/competition")


def emb_memmap(split):
    d = COMP / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def ranges(off, n):
    tot = int(n.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    rep = np.repeat(off - np.concatenate([[0], np.cumsum(n)[:-1]]), n)
    return np.arange(tot, dtype=np.int64) + rep


def load_side(split, sel=None, with_vec=True):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    H = f["H"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy().astype(np.int64)
    M = f["M"].to_numpy().astype(np.int64); co = f["cand_off"].to_numpy().astype(np.int64)
    if sel is None:
        sel = np.arange(len(f))
    h = H[sel]; m = M[sel]
    hr = ranges(ho[sel], h); cr = ranges(co[sel], m)
    A = lambda n, r: np.asarray(np.load(FLAT / split / f"{n}.npy", mmap_mode="r")[r])  # noqa: E731
    out = dict(H=h, hoff=np.concatenate([[0], np.cumsum(h)[:-1]]).astype(np.int64), M=m,
               coff=np.concatenate([[0], np.cumsum(m)[:-1]]).astype(np.int64))
    out["h_emb"] = A("hist_emb_idx", hr); out["h_cat"] = A("hist_category_id", hr).astype(np.int16)
    out["h_type"] = A("hist_event_type", hr).astype(np.int16); out["h_src"] = A("hist_event_source", hr).astype(np.int16)
    out["h_pos"] = A("hist_hpos", hr).astype(np.int64)
    out["c_emb"] = A("cand_emb_idx", cr); out["c_cat"] = A("cand_category_id", cr).astype(np.int16)
    if with_vec:
        u = np.unique(np.concatenate([out["h_emb"][out["h_emb"] >= 0], out["c_emb"][out["c_emb"] >= 0]]))
        mm = emb_memmap(split)
        V = np.empty((len(u), mm.shape[1]), np.float32)
        for s in range(0, len(u), 1_000_000):
            V[s:s + 1_000_000] = mm[u[s:s + 1_000_000]].astype(np.float32)
        nrm = np.linalg.norm(V, axis=1, keepdims=True)
        V /= np.where(nrm > 0, nrm, 1.0)
        out["V"] = V
        out["h_loc"] = np.where(out["h_emb"] >= 0, np.searchsorted(u, out["h_emb"]), -1).astype(np.int64)
        out["c_loc"] = np.where(out["c_emb"] >= 0, np.searchsorted(u, out["c_emb"]), -1).astype(np.int64)
    return out
