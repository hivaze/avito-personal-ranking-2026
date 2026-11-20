"""N_taste: item taste profiles learned from TRAIN labels (new ID-level, personalised information).

For every item j that appears as a val/test candidate, aggregate over its TRAIN impressions (feed f, target t_f, production
position p_f; train ends before val starts, so all of it is past for val and test):
  n_j = #impressions, c_j = #clicks, e_j = sum of expected clicks ctr(p_f) (global train CTR by production position),
  S_j = sum_f v_f, C_j = sum_{f clicked} v_f, E_j = sum_f ctr(p_f) v_f,
for two unit user vectors v: u = user_emb (real-time two-tower), h = recency-weighted mean (half-life 5 events) of the last 20
history item embeddings.
Features for a val/test candidate (feed g, item j), with v_g the same vector for the requesting feed:
  tp_n, tp_c, tp_coec = (c + 1) / (e + 1)
  tp_cosC_v = cos(v_g, C_j)   similarity to the centroid of the users who clicked j      (NaN if c_j = 0)
  tp_cosS_v = cos(v_g, S_j)   similarity to the centroid of the users j was shown to     (NaN if n_j = 0)
  tp_dC_v   = cosC - cosS
  tp_sur_v  = (v_g.C_j - (c_j / e_j) v_g.E_j) / sqrt(c_j + 1)   click surplus among similar users vs a position-debiased null
Computed identically for val and test (train labels only). No same-user later information (train precedes val and test).
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np, polars as pl
import scipy.sparse as sp
from pathlib import Path

F = Path("<repo>/data/flat")
E = Path("<repo>/data/competition")
OUT = Path("<repo>/eda/errors/synthesis/derived")
HL, K = 5.0, 20
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(n > 0, x / np.maximum(n, 1e-12), 0).astype(np.float32)


def hist_vec(split):
    f = pl.read_parquet(F / split / "feeds.parquet")
    nF = len(f)
    hpos = np.load(F / split / "hist_hpos.npy", mmap_mode="r")
    hfidx = np.load(F / split / "hist_fidx.npy", mmap_mode="r")
    hemb = np.load(F / split / "hist_emb_idx.npy", mmap_mode="r")
    meta = E / split / "item_embeddings" / "data.memmap"
    nrow = os.path.getsize(meta) // 128
    emb = np.memmap(meta, dtype=np.int8, mode="r", shape=(nrow, 128))
    out = np.zeros((nF, 128), np.float32)
    CH = 50_000_000
    for s in range(0, len(hpos), CH):
        hp = np.asarray(hpos[s:s + CH]); sel = np.nonzero(hp < K)[0]
        fi = np.asarray(hfidx[s:s + CH])[sel]; ei = np.asarray(hemb[s:s + CH])[sel]; w = 0.5 ** (hp[sel] / HL)
        ok = ei >= 0
        fi, ei, w = fi[ok], ei[ok], w[ok].astype(np.float32)
        order = np.argsort(ei, kind="stable")  # sequential memmap access
        vec = emb[ei[order]].astype(np.float32) * w[order, None]
        if len(fi):
            _segsum(out, fi[order], vec)
        log(split, "hist chunk", s // CH, len(sel))
    return unit(out)


def _segsum(out, fi, vec):
    o = np.argsort(fi, kind="stable"); fi = fi[o]; vec = vec[o]
    b = np.r_[0, np.nonzero(np.diff(fi))[0] + 1]
    out[fi[b]] += np.add.reduceat(vec, b, axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    need = np.unique(np.concatenate([np.load(F / s / "cand_item_id.npy") for s in ("val", "test")]))
    log("needed items", len(need))
    it = np.load(F / "train/cand_item_id.npy")
    ix = np.searchsorted(need, it); ix[ix >= len(need)] = 0
    m = need[ix] == it
    del it
    rows = np.nonzero(m)[0]; ix = ix[rows]
    log("train rows on needed items", len(rows), f"({m.mean():.3f})")
    t = np.load(F / "train/cand_target.npy")
    pos = np.load(F / "train/cand_pos.npy")
    ctr = np.bincount(pos.astype(np.int64), weights=t.astype(np.float64)) / np.maximum(np.bincount(pos.astype(np.int64)), 1)
    tt = t[rows].astype(np.float32); ee = ctr[pos[rows]].astype(np.float32)
    del t, pos
    fidx = np.load(F / "train/cand_fidx.npy")[rows]
    nF = len(pl.read_parquet(F / "train/feeds.parquet", columns=["fidx"]))
    nI = len(need)
    A1 = sp.csr_matrix((np.ones(len(rows), np.float32), (ix, fidx)), shape=(nI, nF))
    Ac = sp.csr_matrix((tt, (ix, fidx)), shape=(nI, nF))
    Ae = sp.csr_matrix((ee, (ix, fidx)), shape=(nI, nF))
    n = np.asarray(A1.sum(1)).ravel(); c = np.asarray(Ac.sum(1)).ravel(); e = np.asarray(Ae.sum(1)).ravel()
    log("aggregates: items with impr", (n > 0).sum(), "with click", (c > 0).sum())
    vecs_tr = {"u": unit(np.load(F / "train/user_emb.npy").astype(np.float32)), "h": hist_vec("train")}
    prof = {}
    for k, V in vecs_tr.items():
        prof[k] = (A1 @ V, Ac @ V, Ae @ V)
        log("profiles", k)
    del vecs_tr
    for split in ("val", "test"):
        cit = np.load(F / split / "cand_item_id.npy"); cf = np.load(F / split / "cand_fidx.npy")
        j = np.searchsorted(need, cit)
        o = OUT / split; o.mkdir(parents=True, exist_ok=True)
        nj, cj, ej = n[j], c[j], e[j]
        np.save(o / "tp_n.npy", nj.astype(np.float32)); np.save(o / "tp_c.npy", cj.astype(np.float32))
        np.save(o / "tp_coec.npy", ((cj + 1) / (ej + 1)).astype(np.float32))
        vg = {"u": unit(np.load(F / split / "user_emb.npy").astype(np.float32)), "h": hist_vec(split)}
        for k in ("u", "h"):
            S, C, Ee = prof[k]
            v = vg[k][cf]
            dS = np.einsum("ij,ij->i", v, S[j]); dC = np.einsum("ij,ij->i", v, C[j]); dE = np.einsum("ij,ij->i", v, Ee[j])
            nS = np.linalg.norm(S[j], axis=1); nC = np.linalg.norm(C[j], axis=1)
            vz = np.linalg.norm(v, axis=1) == 0
            cosS = np.where((nS > 0) & ~vz, dS / np.maximum(nS, 1e-9), np.nan)
            cosC = np.where((nC > 0) & ~vz, dC / np.maximum(nC, 1e-9), np.nan)
            sur = np.where((nj > 0) & ~vz, (dC - cj / np.maximum(ej, 1e-6) * dE) / np.sqrt(cj + 1), np.nan)
            for nm, x in (("cosC", cosC), ("cosS", cosS), ("dC", cosC - cosS), ("sur", sur)):
                np.save(o / f"tp_{nm}_{k}.npy", x.astype(np.float32))
            log(split, k, "cosC nan share", np.isnan(cosC).mean().round(3), "sur nan share", np.isnan(sur).mean().round(3))


if __name__ == "__main__":
    main()
