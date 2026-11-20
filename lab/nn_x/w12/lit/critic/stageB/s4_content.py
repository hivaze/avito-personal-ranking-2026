"""int_gnn s4: can a CONTENT-based (inductive) graph reach the cold candidates?  CPU only (torch on CPU for BLAS), read-only.

(a) WITHIN FEED, all val / test feeds: for every cold candidate (stored LUO count == 0 = no history interaction in any split)
    the highest cosine to a WARM candidate of the same feed (what candidate-set attention / ctx_nd can use at inference).
(b) CATALOGUE, row sample: nearest warm TRAIN item by cosine, exact brute force. Bank = every train-history item with an
    embedding and train-graph degree >= 3 distinct users (segment A: degree >= 10, segment B: degree 3-9).
    Queries per group (val / public / private): NQ random cold rows + NQ random warm rows (train degree >= 3; the item itself
    is removed from its own neighbour list). Cold rows are by construction not in the bank.
(c) content -> collaborative path: does a content-near warm neighbour (cos >= 0.90 / 0.95; best neighbour of each bank chunk,
    top 5) have a 1-hop link (shared user node, own node skipped) with the query feed's own history? For the warm control
    rows the same is computed for the item ITSELF, which gives the fidelity of the content-imputed link.
usage: python s4_content.py NQ"""
import os
import sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import time
from pathlib import Path
import numba as nb
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g_common import build_graph, log, FLAT, FEATS, BOUND  # noqa: E402

COMP = Path("<repo>/data/competition")
NT = 12
torch.set_num_threads(NT)


def table(split):
    d = COMP / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


@nb.njit(parallel=True, cache=True)
def within_feed(coff, M, cemb, tab, luo, out):
    """out[r] = max cosine of candidate r to a WARM (luo > 0) other candidate of its feed (-2 if none / no embedding)."""
    D = tab.shape[1]
    for f in nb.prange(len(M)):
        o = coff[f]
        m = M[f]
        V = np.zeros((m, D), np.float32)
        ok = np.zeros(m, np.bool_)
        for i in range(m):
            e = cemb[o + i]
            if e < 0:
                continue
            v = tab[e].astype(np.float32)
            n = np.sqrt((v * v).sum())
            if n > 0:
                V[i] = v / n
                ok[i] = True
        for i in range(m):
            best = -2.0
            if ok[i]:
                for j in range(m):
                    if j != i and ok[j] and luo[o + j] > 0:
                        s = 0.0
                        for d in range(D):
                            s += V[i, d] * V[j, d]
                        if s > best:
                            best = s
            out[o + i] = best


@nb.njit(cache=True)
def item_emb_row(items, embs, G):
    out = np.full(G, -1, np.int32)
    for e in range(len(items)):
        if embs[e] >= 0:
            out[items[e]] = embs[e]
    return out


@nb.njit(cache=True)
def link_check(qfeed, qitem, nbr, nnb, hoff, H, own, hist, i_ptr, i_users, n_nodes, out):
    """out[q, 0] = the query item itself has a 1-hop link with its feed's history; out[q, 1 + k] = neighbour k has one."""
    ustamp = np.zeros(n_nodes, np.int32)
    for q in range(len(qfeed)):
        f = qfeed[q]
        ep = q + 1
        o = own[f]
        for j in range(H[f]):
            h = hist[hoff[f] + j]
            for e in range(i_ptr[h], i_ptr[h + 1]):
                u = i_users[e]
                if u != o:
                    ustamp[u] = ep
        c = qitem[q]
        for e in range(i_ptr[c], i_ptr[c + 1]):
            u = i_users[e]
            if u != o and ustamp[u] == ep:
                out[q, 0] = 1
                break
        for k in range(nnb[q]):
            w = nbr[q, k]
            for e in range(i_ptr[w], i_ptr[w + 1]):
                u = i_users[e]
                if u != o and ustamp[u] == ep:
                    out[q, 1 + k] = 1
                    break


def main():
    NQ = int(sys.argv[1])
    nb.set_num_threads(NT)
    g = build_graph("train")
    feeds = g["feeds"]
    G = g["G"]
    i_ptr, i_users = g["i_ptr"], g["i_users"]
    deg_item = np.diff(i_ptr)
    tts = feeds["test"]["timestamp"].to_numpy()
    groups = {"val": ("val", np.ones(len(feeds["val"]), bool)), "public": ("test", tts < BOUND), "private": ("test", tts >= BOUND)}

    # ---------------- (a) within feed ----------------
    S = {}
    for split in ["val", "test"]:
        f = feeds[split]
        M = f["M"].to_numpy().astype(np.int64)
        coff = f["cand_off"].to_numpy().astype(np.int64)
        cemb = np.load(FLAT / split / "cand_emb_idx.npy").astype(np.int64)
        luo = np.asarray(np.load(FEATS / split / "it_hist_all_luo.npy"), np.float64)
        tab = np.ascontiguousarray(table(split))
        wf = np.zeros(len(cemb), np.float32)
        within_feed(coff, M, cemb, tab, luo, wf)
        S[split] = dict(M=M, coff=coff, cemb=cemb, luo=luo, tab=tab, wf=wf, cand=g["cands"][split],
                        fid=np.repeat(np.arange(len(M)), M))
        for gname, (sp, fm) in groups.items():
            if sp != split:
                continue
            rm = np.repeat(fm, M)
            cold = rm & (luo == 0)
            warmr = rm & (luo > 0)
            x = wf[cold]
            xw = wf[warmr]
            log(f"[within feed] {gname}: cold rows {cold.sum()} ({cold.sum() / rm.sum():.4f} of rows) | cold with a warm same-feed candidate at cos "
                f">=0.80 {np.mean(x >= 0.80):.4f} >=0.90 {np.mean(x >= 0.90):.4f} >=0.95 {np.mean(x >= 0.95):.4f} | median max cos {np.median(x):.3f} "
                f"(p10 {np.quantile(x, 0.1):.3f}, p90 {np.quantile(x, 0.9):.3f}) | no warm candidate with an embedding in the feed {np.mean(x < -1):.4f} | "
                f"control, warm rows to another warm row: >=0.80 {np.mean(xw >= 0.80):.4f} >=0.90 {np.mean(xw >= 0.90):.4f} >=0.95 {np.mean(xw >= 0.95):.4f}, median {np.median(xw):.3f}")

    # ---------------- (b) bank ----------------
    it = np.load(FLAT / "train/hist_item_id.npy")
    em = np.load(FLAT / "train/hist_emb_idx.npy")
    emb_row = item_emb_row(it, em, G)
    del it, em
    in_bank = (deg_item >= 3) & (emb_row >= 0)
    log(f"train-history items: degree>=3 {int((deg_item >= 3).sum())}, with an embedding {int(in_bank.sum())} ({in_bank.sum() / max((deg_item >= 3).sum(), 1):.4f}); "
        f"degree>=10 with an embedding {int(((deg_item >= 10) & (emb_row >= 0)).sum())}; any degree with an embedding {int(((deg_item >= 1) & (emb_row >= 0)).sum())} of {int((deg_item >= 1).sum())}")
    bank_items = np.nonzero(in_bank)[0]
    rng = np.random.default_rng(20261115)
    segA = bank_items[deg_item[bank_items] >= 10]
    segB = bank_items[deg_item[bank_items] < 10]
    bank_items = np.concatenate([rng.permutation(segA), rng.permutation(segB)])
    nA = len(segA)
    NB = len(bank_items)
    rows = emb_row[bank_items].astype(np.int64)
    bank = np.empty((NB, 128), np.int8)
    order = np.argsort(rows)
    srows = rows[order]
    tpath = COMP / "train/item_embeddings/data.memmap"
    ntab = json.loads((COMP / "train/item_embeddings/data.meta").read_text())["shape"][0]
    blk = 4_000_000
    for a in range(0, ntab, blk):
        b = min(a + blk, ntab)
        lo, hi = np.searchsorted(srows, [a, b])
        if hi > lo:
            buf = np.fromfile(tpath, dtype=np.int8, count=(b - a) * 128, offset=a * 128).reshape(b - a, 128)
            bank[order[lo:hi]] = buf[srows[lo:hi] - a]
            del buf
    bnorm = np.sqrt((bank.astype(np.float32) ** 2).sum(1))
    log(f"bank built: {NB} items ({nA} with degree >= 10), zero-norm rows {int((bnorm == 0).sum())}")
    pos_in_bank = np.full(G, -1, np.int64)
    pos_in_bank[bank_items] = np.arange(NB)

    # ---------------- queries ----------------
    Q = []
    for gname, (split, fm) in groups.items():
        s = S[split]
        rm = np.repeat(fm, s["M"]) & (s["cemb"] >= 0)
        cold = np.nonzero(rm & (s["luo"] == 0))[0]
        warm = np.nonzero(rm & (pos_in_bank[s["cand"]] >= 0))[0]
        for kind, pool in [("cold", cold), ("warm", warm)]:
            sel = np.sort(rng.choice(pool, min(NQ, len(pool)), replace=False))
            Q.append((gname, split, kind, sel))
    qvec = np.concatenate([S[sp]["tab"][S[sp]["cemb"][sel]] for _, sp, _, sel in Q]).astype(np.float32)
    qvec /= np.maximum(np.sqrt((qvec ** 2).sum(1, keepdims=True)), 1e-6)
    qitem = np.concatenate([S[sp]["cand"][sel] for _, sp, _, sel in Q])
    qself = pos_in_bank[qitem]
    nq = len(qitem)
    log(f"queries {nq}")
    CH = 100_000
    nch = (NB + CH - 1) // CH
    best_v = np.full((nq, nch), -2.0, np.float32)   # best neighbour of every bank chunk
    best_i = np.zeros((nq, nch), np.int64)
    self_cos = np.full(nq, np.nan, np.float32)
    qt = torch.from_numpy(qvec)
    QB = 4000
    t1 = time.time()
    for ci, a in enumerate(range(0, NB, CH)):
        b = min(a + CH, NB)
        ck = torch.from_numpy(bank[a:b].astype(np.float32))
        ck /= torch.from_numpy(np.maximum(bnorm[a:b], 1e-6))[:, None]
        ckt = ck.T.contiguous()
        inch = np.nonzero((qself >= a) & (qself < b))[0]
        for q0 in range(0, nq, QB):
            sim = qt[q0:q0 + QB] @ ckt
            mine = inch[(inch >= q0) & (inch < q0 + QB)]
            if len(mine):
                self_cos[mine] = sim[mine - q0, qself[mine] - a].numpy()
                sim[mine - q0, qself[mine] - a] = -2.0
            v, i = sim.max(dim=1)
            best_v[q0:q0 + QB, ci] = v.numpy()
            best_i[q0:q0 + QB, ci] = i.numpy() + a
        if ci % 40 == 0:
            log(f"chunk {ci}/{nch} ({time.time() - t1:.0f}s)")
    chA = np.arange(nch) * CH + CH <= nA       # chunks entirely in segment A
    full_A = best_v[:, chA].max(1) if chA.any() else np.full(nq, -2.0)
    # exact split of the boundary chunk is not needed: segment A statistic uses whole-A chunks only (lower bound, <= 1 chunk short)
    top1 = best_v.max(1)
    k5 = np.argsort(-best_v, axis=1)[:, :5]
    nv = np.take_along_axis(best_v, k5, 1)
    ni = bank_items[np.take_along_axis(best_i, k5, 1)]
    np.savez(HERE / "s4_knn.npz", top1=top1, topA=full_A, nv=nv, ni=ni, qitem=qitem, self_cos=self_cos)

    # ---------------- (c) links ----------------
    off = 0
    for gname, split, kind, sel in Q:
        n = len(sel)
        sl = slice(off, off + n)
        off += n
        s = S[split]
        f = feeds[split]
        qfeed = s["fid"][sel].astype(np.int64)
        hist = np.load(FLAT / split / "hist_item_id.npy").astype(np.int64)
        hoff = f["hist_off"].to_numpy().astype(np.int64)
        Hh = f["H"].to_numpy().astype(np.int64)
        own = g["node_of_feed"][split].astype(np.int64)
        t = top1[sl]
        line = (f"[catalogue] {gname} {kind}: {n} rows | nearest warm train item (degree>=3 bank): cos >=0.80 {np.mean(t >= 0.80):.4f} >=0.90 {np.mean(t >= 0.90):.4f} "
                f">=0.95 {np.mean(t >= 0.95):.4f} >=0.99 {np.mean(t >= 0.99):.4f}; median {np.median(t):.3f} (p10 {np.quantile(t, 0.1):.3f}, p90 {np.quantile(t, 0.9):.3f}) | "
                f"degree>=10 bank only: >=0.80 {np.mean(full_A[sl] >= 0.80):.4f} >=0.90 {np.mean(full_A[sl] >= 0.90):.4f} >=0.95 {np.mean(full_A[sl] >= 0.95):.4f}; median {np.median(full_A[sl]):.3f}")
        if kind == "warm":
            sc = self_cos[sl]
            line += f" | same item, split table vs train table: cos mean {np.nanmean(sc):.4f}, share >=0.999 {np.nanmean(sc >= 0.999):.4f}"
        log(line)
        for thr in [0.90, 0.95]:
            nnb = (nv[sl] >= thr).sum(1).astype(np.int64)
            out = np.zeros((n, 6), np.int8)
            link_check(qfeed, qitem[sl].astype(np.int64), ni[sl].astype(np.int64), nnb, hoff, Hh, own, hist, i_ptr, i_users, g["n_nodes"], out)
            imp = out[:, 1:].max(1)
            line = (f"[content->collab] {gname} {kind}, neighbours at cos >= {thr}: rows with >=1 such neighbour {np.mean(nnb >= 1):.4f} | "
                    f"rows where such a neighbour is 1-hop linked to the feed's history {np.mean(imp):.4f}")
            if kind == "warm":
                own_l = out[:, 0]
                line += (f" | own 1-hop link {np.mean(own_l):.4f} | P(imputed link | own link) {np.mean(imp[own_l == 1]):.4f} | P(own link | imputed link) "
                         f"{np.mean(own_l[imp == 1]) if imp.sum() else float('nan'):.4f} | P(own link | no imputed link) {np.mean(own_l[imp == 0]):.4f}")
            log(line)
    log("DONE")


if __name__ == "__main__":
    main()
