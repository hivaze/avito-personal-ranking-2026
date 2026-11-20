"""int_gnn s2: user-item interaction graph from the flat history tables; degree and hop reach of val / test candidates.
CPU only, 12 threads, read-only on the project; writes only into this directory.

usage: python s2_graph.py MODE NFEEDS       MODE = train | all
Graph: bipartite (user node, item). A user node = one non-zero user_id (union of ALL its history snapshots in the included
splits) or one user-0 feed (user 0 is a pooled bucket: each of its feeds is its own node). Edge = the item occurs in a history
of that node (any event type), de-duplicated. MODE train: train histories only. MODE all: train + val + test histories
(the most generous transductive graph: every label-free history on disk, including other users' later snapshots).
For a query feed (val / test) its OWN user node is skipped everywhere (no same-user path; the leak rule forbids the same
user's later snapshots, and the user's own history enters as the start set H anyway).

Per candidate c of a query feed with history item set H:
  deg       = number of user nodes holding c (any user);  deg_x = the same without the own node
  hop1 cnt  = number of user nodes (not own) that hold c AND at least one item of H   (path c - u - h; capped at 10)
  hop2 cnt  = number of user nodes u (not own) that hold c AND some item n != c with n in I1, where
              I1 = all items held by U1 = users holding at least one item of H   (path c - u - n - u' - h; capped at 10)
  hop2 nh   = hop2 through a NON-HUB bridge item n (deg(n) <= 100)
  hop2 scans at most CAPU user nodes of c (first by node id); 'trunc' marks candidates whose scan was cut before a decision.
Train-impression / click counts of the candidate item (train cand rows, any user) are added in MODE train.
"""
import os
import sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
from pathlib import Path
import numba as nb
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
FLAT = Path("<repo>/data/flat")
FEATS = Path("<repo>/data/feats")
BOUND = 1790497920
NT = 12
SH = 27  # item ids < 2^27
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


@nb.njit(parallel=True, cache=True)
def make_keys(item, fidx, node_of_feed, out, off):
    for e in nb.prange(len(item)):
        out[off + e] = (np.int64(node_of_feed[fidx[e]]) << SH) | np.int64(item[e])


@nb.njit(cache=True)
def compact_unique(keys):
    n = len(keys)
    if n == 0:
        return 0
    w = 1
    for i in range(1, n):
        if keys[i] != keys[w - 1]:
            keys[w] = keys[i]
            w += 1
    return w


@nb.njit(cache=True)
def build_csr_csc(keys, n_nodes, G):
    nnz = len(keys)
    mask = (np.int64(1) << SH) - 1
    u_ptr = np.zeros(n_nodes + 1, np.int64)
    i_ptr = np.zeros(G + 1, np.int64)
    u_items = np.empty(nnz, np.int32)
    for e in range(nnz):
        u = keys[e] >> SH
        g = keys[e] & mask
        u_ptr[u + 1] += 1
        i_ptr[g + 1] += 1
        u_items[e] = g
    for u in range(n_nodes):
        u_ptr[u + 1] += u_ptr[u]
    for g in range(G):
        i_ptr[g + 1] += i_ptr[g]
    fill = i_ptr[:-1].copy()
    i_users = np.empty(nnz, np.int32)
    for e in range(nnz):
        u = keys[e] >> SH
        g = keys[e] & mask
        i_users[fill[g]] = u
        fill[g] += 1
    return u_ptr, u_items, i_ptr, i_users


@nb.njit(parallel=True, cache=True)
def deg_rows(cand, cfeed, own_node, i_ptr, i_users, deg, degx):
    for r in nb.prange(len(cand)):
        c = cand[r]
        a = i_ptr[c]
        b = i_ptr[c + 1]
        d = b - a
        deg[r] = d
        o = own_node[cfeed[r]]
        x = d
        if o >= 0 and d > 0:
            lo = a
            hi = b
            while lo < hi:
                m = (lo + hi) >> 1
                if i_users[m] < o:
                    lo = m + 1
                else:
                    hi = m
            if lo < b and i_users[lo] == o:
                x = d - 1
        degx[r] = x


@nb.njit(parallel=True, cache=True)
def reach(sel, coff, M, hoff, H, own_node, cand, hist, u_ptr, u_items, i_ptr, i_users, n_nodes, G, nthr, capu, hub,
          out, roff, finfo):
    for t in nb.prange(nthr):
        ustamp = np.zeros(n_nodes, np.int32)
        ulist = np.empty(n_nodes, np.int32)
        istamp = np.zeros(G, np.uint16)
        ep = 0
        for q in range(t, len(sel), nthr):
            f = sel[q]
            ep += 1
            o = own_node[f]
            ho = hoff[f]
            nu = 0
            # U1 = users (not own) holding at least one history item
            for j in range(H[f]):
                h = hist[ho + j]
                for e in range(i_ptr[h], i_ptr[h + 1]):
                    u = i_users[e]
                    if u != o and ustamp[u] != ep:
                        ustamp[u] = ep
                        ulist[nu] = u
                        nu += 1
            # I1 = items held by U1
            ni = 0
            for k in range(nu):
                u = ulist[k]
                for e in range(u_ptr[u], u_ptr[u + 1]):
                    g = u_items[e]
                    if istamp[g] != ep:
                        istamp[g] = ep
                        ni += 1
            finfo[q, 0] = nu
            finfo[q, 1] = ni
            co = coff[f]
            ro = roff[q]
            for i in range(M[f]):
                c = cand[co + i]
                a = i_ptr[c]
                b = i_ptr[c + 1]
                c1 = 0
                c2 = 0
                nh = 0
                trunc = 0
                # hop 1
                for e in range(a, b):
                    u = i_users[e]
                    if u != o and ustamp[u] == ep:
                        c1 += 1
                        if c1 >= 10:
                            break
                if c1 > 0:
                    nh = 1
                # hop 2 (includes hop 1: a hop-1 user also holds an item of H, which is in I1)
                scanned = 0
                for e in range(a, b):
                    u = i_users[e]
                    if u == o:
                        continue
                    if c2 >= 10 and nh == 1:
                        break
                    if scanned >= capu:
                        trunc = 1
                        break
                    scanned += 1
                    hit = 0
                    for e2 in range(u_ptr[u], u_ptr[u + 1]):
                        n = u_items[e2]
                        if n != c and istamp[n] == ep:
                            hit = 1
                            if nh == 1:
                                break
                            if i_ptr[n + 1] - i_ptr[n] <= hub:
                                nh = 1
                                break
                    if hit == 1 and c2 < 10:
                        c2 += 1
                out[ro + i, 0] = c1
                out[ro + i, 1] = c2
                out[ro + i, 2] = nh
                out[ro + i, 3] = trunc


def main():
    mode = sys.argv[1]
    nfeeds = int(sys.argv[2])
    capu = 3000
    hub = 100
    nb.set_num_threads(NT)
    feeds = {s: pl.read_parquet(FLAT / s / "feeds.parquet") for s in ["train", "val", "test"]}
    splits = ["train"] if mode == "train" else ["train", "val", "test"]
    # ---- user nodes
    nz = np.unique(np.concatenate([feeds[s]["user_id"].to_numpy() for s in splits]))
    nz = nz[nz != 0]
    n_nz = len(nz)
    node_of_feed = {}
    nxt = n_nz
    for s in ["train", "val", "test"]:
        u = feeds[s]["user_id"].to_numpy()
        k = np.searchsorted(nz, u)
        k = np.minimum(k, n_nz - 1)
        node = np.where((u != 0) & (nz[k] == u), k, -1).astype(np.int64)
        if s in splits:  # user-0 feeds of an included split are nodes of their own
            z = np.nonzero(u == 0)[0]
            node[z] = nxt + np.arange(len(z))
            nxt += len(z)
        node_of_feed[s] = node
    n_nodes = nxt
    log(f"mode {mode}: user nodes {n_nodes} ({n_nz} non-zero users + {n_nodes - n_nz} user-0 feeds)")
    # ---- edges
    n_ev = {s: int(feeds[s]["H"].to_numpy().astype(np.int64).sum()) for s in splits}
    keys = np.empty(sum(n_ev.values()), np.int64)
    off = 0
    G = 0
    for s in splits:
        it = np.load(FLAT / s / "hist_item_id.npy")
        fx = np.load(FLAT / s / "hist_fidx.npy")
        assert len(it) == n_ev[s] and it.min() >= 0
        G = max(G, int(it.max()) + 1)
        make_keys(it, fx, node_of_feed[s], keys, off)
        off += len(it)
        del it, fx
    cands = {s: np.load(FLAT / s / "cand_item_id.npy").astype(np.int64) for s in ["val", "test"]}
    G = max(G, int(cands["val"].max()) + 1, int(cands["test"].max()) + 1)
    assert G < (1 << SH)
    log(f"events {len(keys)}; item id space {G}")
    keys.sort()
    nnz = compact_unique(keys)
    log(f"distinct (user node, item) edges {nnz} ({nnz / len(keys):.4f} of events)")
    keys = keys[:nnz]
    u_ptr, u_items, i_ptr, i_users = build_csr_csc(keys, n_nodes, G)
    del keys
    deg_item = np.diff(i_ptr)
    n_items = int((deg_item > 0).sum())
    ulen = np.diff(u_ptr)
    log(f"graph: items with degree >= 1: {n_items}; item degree ==1 {np.mean(deg_item[deg_item > 0] == 1):.4f} "
        f">=2 {np.mean(deg_item[deg_item > 0] >= 2):.4f} >=3 {np.mean(deg_item[deg_item > 0] >= 3):.4f} >=10 {np.mean(deg_item[deg_item > 0] >= 10):.4f} "
        f">=100 {np.mean(deg_item[deg_item > 0] >= 100):.5f} max {deg_item.max()}; "
        f"user-node degree mean {ulen[ulen > 0].mean():.1f} median {np.median(ulen[ulen > 0]):.0f} max {ulen.max()}; nodes with >= 1 edge {int((ulen > 0).sum())}")
    edge_share_hub = deg_item[deg_item > hub].sum() / nnz
    log(f"edges that touch an item with degree > {hub}: {edge_share_hub:.4f}; items with degree > {hub}: {int((deg_item > hub).sum())}")

    extra = {}
    if mode == "train":  # train impressions / clicks per item (any user)
        ci = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
        ct = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
        imp = np.zeros(G, np.int32)
        clk = np.zeros(G, np.int32)
        step = 20_000_000
        for a in range(0, len(ci), step):
            x = np.asarray(ci[a:a + step]).astype(np.int64)
            y = np.asarray(ct[a:a + step]).astype(np.int64)
            ok = x < G
            imp += np.bincount(x[ok], minlength=G).astype(np.int32)
            clk += np.bincount(x[ok], weights=y[ok], minlength=G).astype(np.int32)
        extra = {"imp": imp, "clk": clk}
        log(f"train impressions: items shown {int((imp > 0).sum())}, clicked {int((clk > 0).sum())}; items shown but never in a train history {int(((imp > 0) & (deg_item[:G] == 0)).sum())}")

    rng = np.random.default_rng(20261115)
    tts = feeds["test"]["timestamp"].to_numpy()
    groups = {"val": ("val", np.ones(len(feeds["val"]), bool)), "public": ("test", tts < BOUND), "private": ("test", tts >= BOUND)}
    res = {}
    for split in ["val", "test"]:
        f = feeds[split]
        M = f["M"].to_numpy().astype(np.int64)
        Hh = f["H"].to_numpy().astype(np.int64)
        coff = f["cand_off"].to_numpy().astype(np.int64)
        hoff = f["hist_off"].to_numpy().astype(np.int64)
        own = node_of_feed[split].astype(np.int64)
        cand = cands[split]
        cfeed = np.repeat(np.arange(len(f)), M)
        deg = np.zeros(len(cand), np.int64)
        degx = np.zeros(len(cand), np.int64)
        deg_rows(cand, cfeed, own, i_ptr, i_users, deg, degx)
        hist = np.load(FLAT / split / "hist_item_id.npy").astype(np.int64)
        luo = np.asarray(np.load(FEATS / split / "it_hist_all_luo.npy", mmap_mode="r"))
        res[split] = dict(M=M, deg=deg, degx=degx, cfeed=cfeed, luo=luo)
        for g, (sp, fm) in groups.items():
            if sp != split:
                continue
            rm = np.repeat(fm, M)
            d = deg[rm]; dx = degx[rm]; l = luo[rm]
            line = (f"[deg {mode}] {g}: rows {rm.sum()} | any user: >=1 {np.mean(d >= 1):.4f} >=3 {np.mean(d >= 3):.4f} >=10 {np.mean(d >= 10):.4f} | "
                    f"other users only: >=1 {np.mean(dx >= 1):.4f} >=3 {np.mean(dx >= 3):.4f} >=10 {np.mean(dx >= 10):.4f} | "
                    f"stored LUO==0 {np.mean(l == 0):.4f}; deg_x==0 given LUO==0 {np.mean(dx[l == 0] == 0):.4f}; LUO==0 given deg_x==0 {np.mean(l[dx == 0] == 0):.4f}")
            if extra:
                im = extra["imp"][cand[rm]]; ck = extra["clk"][cand[rm]]
                line += (f" | train impressions >=1 {np.mean(im >= 1):.4f} >=3 {np.mean(im >= 3):.4f} >=10 {np.mean(im >= 10):.4f} | train clicks >=1 {np.mean(ck >= 1):.4f} "
                         f">=3 {np.mean(ck >= 3):.4f} >=10 {np.mean(ck >= 10):.4f} | history OR impression >=1 {np.mean((d >= 1) | (im >= 1)):.4f} | "
                         f"impression but no history {np.mean((d == 0) & (im >= 1)):.4f}")
            log(line)
            # hop reach on a feed sample
            ids = np.nonzero(fm)[0]
            sel = np.sort(rng.choice(ids, min(nfeeds, len(ids)), replace=False)).astype(np.int64)
            Ms = M[sel]
            roff = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64)
            out = np.zeros((int(Ms.sum()), 4), np.int16)
            finfo = np.zeros((len(sel), 2), np.int64)
            t1 = time.time()
            reach(sel, coff, M, hoff, Hh, own, cand, hist, u_ptr, u_items, i_ptr, i_users, n_nodes, G, NT, capu, hub, out, roff, finfo)
            rows = np.concatenate([np.arange(coff[x], coff[x] + M[x]) for x in sel])
            dxs = degx[rows]; ls = luo[rows]
            c1 = out[:, 0]; c2 = out[:, 1]; nh = out[:, 2]; tr = out[:, 3]
            # per-feed clustered SE of the main shares
            fid = np.repeat(np.arange(len(sel)), Ms)

            def sh(x):
                x = x.astype(np.float64)
                p = x.mean()
                num = np.bincount(fid, weights=x - p, minlength=len(sel))
                se = np.sqrt((num ** 2).sum()) / len(x)
                return f"{p:.4f} (se {se:.4f})"
            hist_has = np.array([np.mean(deg_item[hist[hoff[x]:hoff[x] + Hh[x]]] > 0) if Hh[x] > 0 else np.nan for x in sel])
            log(f"[reach {mode}] {g}: {len(sel)} feeds, {len(rows)} rows, {time.time() - t1:.0f}s | deg_x>=1 {sh(dxs >= 1)} | "
                f"hop1 >=1 {sh(c1 >= 1)} >=3 {sh(c1 >= 3)} >=10 {sh(c1 >= 10)} | hop2 >=1 {sh(c2 >= 1)} >=3 {sh(c2 >= 3)} >=10 {sh(c2 >= 10)} | "
                f"hop2 via non-hub bridge (deg<={hub}) {sh(nh >= 1)} | undecided by the {capu}-user scan cap {np.mean((tr == 1) & ((c2 < 1) | (nh == 0))):.4f} | "
                f"given deg_x>=1: hop1 {np.mean(c1[dxs >= 1] >= 1):.4f} hop2 {np.mean(c2[dxs >= 1] >= 1):.4f} hop2-nonhub {np.mean(nh[dxs >= 1] >= 1):.4f} | "
                f"stored-cold rows (LUO==0): hop1 {np.mean(c1[ls == 0] >= 1):.4f} hop2 {np.mean(c2[ls == 0] >= 1):.4f} | "
                f"feeds: median |U1| {np.median(finfo[:, 0]):.0f} ({np.median(finfo[:, 0]) / n_nodes:.4f} of nodes), median |I1| {np.median(finfo[:, 1]):.0f} "
                f"({np.median(finfo[:, 1]) / n_items:.4f} of items), feeds with |U1|==0 {np.mean(finfo[:, 0] == 0):.4f} | history events with item degree >=1 (mean over feeds) {np.nanmean(hist_has):.4f}")
            np.savez(HERE / f"reach_{mode}_{g}.npz", sel=sel, rows=rows, out=out, finfo=finfo, degx=dxs, deg=deg[rows], luo=ls)
    log("DONE")


if __name__ == "__main__":
    main()
