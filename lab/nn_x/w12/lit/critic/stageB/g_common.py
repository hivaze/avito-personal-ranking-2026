"""int_gnn shared graph builder (same construction as s2_graph.py; see its docstring). CPU only."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
from pathlib import Path
import numpy as np
import polars as pl
from s2_graph import make_keys, compact_unique, build_csr_csc, SH

FLAT = Path("<repo>/data/flat")
FEATS = Path("<repo>/data/feats")
BOUND = 1790497920
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def build_graph(mode):
    feeds = {s: pl.read_parquet(FLAT / s / "feeds.parquet") for s in ["train", "val", "test"]}
    splits = ["train"] if mode == "train" else ["train", "val", "test"]
    nz = np.unique(np.concatenate([feeds[s]["user_id"].to_numpy() for s in splits]))
    nz = nz[nz != 0]
    n_nz = len(nz)
    node_of_feed = {}
    nxt = n_nz
    for s in ["train", "val", "test"]:
        u = feeds[s]["user_id"].to_numpy()
        k = np.minimum(np.searchsorted(nz, u), n_nz - 1)
        node = np.where((u != 0) & (nz[k] == u), k, -1).astype(np.int64)
        if s in splits:
            z = np.nonzero(u == 0)[0]
            node[z] = nxt + np.arange(len(z))
            nxt += len(z)
        node_of_feed[s] = node
    n_nodes = nxt
    n_ev = {s: int(feeds[s]["H"].to_numpy().astype(np.int64).sum()) for s in splits}
    keys = np.empty(sum(n_ev.values()), np.int64)
    off = 0
    G = 0
    for s in splits:
        it = np.load(FLAT / s / "hist_item_id.npy")
        fx = np.load(FLAT / s / "hist_fidx.npy")
        G = max(G, int(it.max()) + 1)
        make_keys(it, fx, node_of_feed[s], keys, off)
        off += len(it)
        del it, fx
    cands = {s: np.load(FLAT / s / "cand_item_id.npy").astype(np.int64) for s in ["val", "test"]}
    G = max(G, int(cands["val"].max()) + 1, int(cands["test"].max()) + 1)
    assert G < (1 << SH)
    keys.sort()
    nnz = compact_unique(keys)
    keys = keys[:nnz]
    u_ptr, u_items, i_ptr, i_users = build_csr_csc(keys, n_nodes, G)
    del keys
    log(f"graph {mode}: {n_nodes} user nodes, {nnz} edges, item id space {G}")
    return dict(feeds=feeds, node_of_feed=node_of_feed, n_nodes=n_nodes, G=G, u_ptr=u_ptr, u_items=u_items, i_ptr=i_ptr,
                i_users=i_users, cands=cands)
