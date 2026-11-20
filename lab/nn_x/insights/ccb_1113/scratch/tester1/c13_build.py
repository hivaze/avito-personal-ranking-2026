"""C13 pending-favourite (saved, not contacted). Own-history only (B10), label-free.
Per feed, per distinct history item_id (256 window = the stored history): n_e4 and any e2/e3/e5/e6/e8 event on that item.
P_u = items with odd n_e4 and no contact/delivery event. p1 = max cos(c, P_u items with a vector) (NaN if none / no candidate
vector); p2 = p1 - max cos(c, all e4 items with a vector); p3 = |P_u items in cat(c)| / (|P_u| + 1).
Outputs out/c13_{p1,p2,p3}_{val,hold}.npy (val flat order; holdout = preds/holdout order)."""
import os, sys, json, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np, polars as pl, numba as nb
from pathlib import Path
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"
OUT = Path(__file__).resolve().parent / "out"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def table(split):
    d = ROOT / "data/competition" / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.fromfile(d / "data.memmap", dtype=np.int8).reshape(meta["shape"])


@nb.njit(parallel=True)
def kernel(hoff, H, coff, M, ooff, h_item, h_et, h_emb, h_cat, c_emb, c_cat, E, p1, p2, p3, stats):
    for f in nb.prange(len(H)):
        h0 = hoff[f]; n = H[f]
        items = h_item[h0:h0 + n]
        order = np.argsort(items, kind="mergesort")
        # distinct items
        uitem = np.empty(n, np.int64); ne4 = np.zeros(n, np.int64); cont = np.zeros(n, np.bool_)
        uemb = np.full(n, -1, np.int64); ucat = np.full(n, -1, np.int64)
        nu = -1; last = -1
        for t in range(n):
            j = h0 + order[t]
            it = h_item[j]
            if t == 0 or it != last:
                nu += 1; uitem[nu] = it; last = it
            et = h_et[j]
            if et == 4:
                ne4[nu] += 1
            if et == 2 or et == 3 or et == 5 or et == 6 or et == 8:
                cont[nu] = True
            if uemb[nu] < 0 and h_emb[j] >= 0:
                uemb[nu] = h_emb[j]
            if ucat[nu] < 0:
                ucat[nu] = h_cat[j]
        nu += 1
        # sets
        nP = 0; nE = 0
        Pv = np.zeros((nu, 128), np.float32); Ev = np.zeros((nu, 128), np.float32)
        Pcat = np.zeros(64, np.int64); nPall = 0
        for u in range(nu):
            if ne4[u] >= 1 and uemb[u] >= 0:
                s = 0.0
                for d in range(128):
                    v = np.float32(E[uemb[u], d]); Ev[nE, d] = v; s += v * v
                s = np.sqrt(s)
                if s > 0:
                    for d in range(128):
                        Ev[nE, d] /= s
                    nE += 1
            if ne4[u] % 2 == 1 and not cont[u]:
                nPall += 1
                if ucat[u] >= 0:
                    Pcat[ucat[u]] += 1
                if uemb[u] >= 0:
                    s = 0.0
                    for d in range(128):
                        v = np.float32(E[uemb[u], d]); Pv[nP, d] = v; s += v * v
                    s = np.sqrt(s)
                    if s > 0:
                        for d in range(128):
                            Pv[nP, d] /= s
                        nP += 1
        stats[f, 0] = nPall; stats[f, 1] = nP; stats[f, 2] = nE
        c0 = coff[f]; o0 = ooff[f]
        cv = np.zeros(128, np.float32)
        for k in range(M[f]):
            j = c0 + k; o = o0 + k
            cc = c_cat[j]
            p3[o] = (Pcat[cc] if cc >= 0 else 0) / (nPall + 1.0)
            e = c_emb[j]
            if nP == 0 or e < 0:
                p1[o] = np.nan; p2[o] = np.nan
                continue
            s = 0.0
            for d in range(128):
                v = np.float32(E[e, d]); cv[d] = v; s += v * v
            s = np.sqrt(s)
            if s == 0:
                p1[o] = np.nan; p2[o] = np.nan
                continue
            best = -2.0
            for q in range(nP):
                t = 0.0
                for d in range(128):
                    t += cv[d] * Pv[q, d]
                if t > best:
                    best = t
            bestE = -2.0
            for q in range(nE):
                t = 0.0
                for d in range(128):
                    t += cv[d] * Ev[q, d]
                if t > bestE:
                    bestE = t
            p1[o] = best / s; p2[o] = best / s - bestE / s


def build(split, feeds_idx, tag):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); coff = f["cand_off"].to_numpy().astype(np.int64)
    H = f["H"].to_numpy().astype(np.int64); hoff = f["hist_off"].to_numpy().astype(np.int64)
    if feeds_idx is not None:
        M, coff, H, hoff = M[feeds_idx], coff[feeds_idx], H[feeds_idx], hoff[feeds_idx]
    ooff = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    A = {n: np.load(FLAT / split / f"{n}.npy", mmap_mode="r") for n in
         ["hist_item_id", "hist_event_type", "hist_emb_idx", "hist_category_id", "cand_emb_idx", "cand_category_id"]}
    A = {k: np.asarray(v) for k, v in A.items()}
    E = table(split)
    N = int(M.sum())
    p1 = np.full(N, np.nan); p2 = np.full(N, np.nan); p3 = np.zeros(N)
    stats = np.zeros((len(M), 3), np.int64)
    kernel(hoff, H, coff, M, ooff, A["hist_item_id"].astype(np.int64), A["hist_event_type"].astype(np.int64),
           A["hist_emb_idx"].astype(np.int64), A["hist_category_id"].astype(np.int64), A["cand_emb_idx"].astype(np.int64),
           A["cand_category_id"].astype(np.int64), E, p1, p2, p3, stats)
    for n, x in (("p1", p1), ("p2", p2), ("p3", p3)):
        np.save(OUT / f"c13_{n}_{tag}.npy", x.astype(np.float32))
    log(f"{tag}: feeds {len(M)} rows {N} | feeds with |P_u|>=1 {np.mean(stats[:, 0] > 0):.4f} (with vector {np.mean(stats[:, 1] > 0):.4f})"
        f" mean |P_u| {stats[:, 0].mean():.2f} mean |e4 items| {stats[:, 2].mean():.2f} | p1 finite {np.isfinite(p1).mean():.4f}"
        f" p1 mean {np.nanmean(p1):.4f} p2 mean {np.nanmean(p2):.4f} p3>0 {np.mean(p3 > 0):.4f}")


build("val", None, "val")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h import holdout_ids
hold = holdout_ids()[0]
build("train", hold, "hold")
log("DONE")
