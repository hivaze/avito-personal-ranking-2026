"""Learned next-click embedding predictor (linear "where is the user heading" model).

Fit (ridge, closed form) on de-duplicated train user sequences: target = embedding of an event_type-1 event at
hpos j (j = the newest type-1 event with hpos in 1..12), inputs = pooled unit vectors of the events OLDER than j
(re-indexed so the event right before the target has recency 0):
  decayed means with half-lives 1, 3, 10, 40 over all events; decayed mean (hl 5) over type-1 events;
  decayed mean (hl 10) over non-type-1 events; deep events (5,6,8,9; hl 30); repeated items (hl 30)  -> 8 x 128.
For a feed the same pooled vectors over its full history give y_hat = X W; features:
  seq_nxt_cos   cos(candidate, y_hat)
The model for val/test is fitted on the all-train source; train rows use a model fitted on the source of days
< Sep 13 for rows on days >= Sep 13 and a model fitted on sources of days >= Sep 13 only... (see run()).
"""
import sys
import numba as nb
import numpy as np
from seq_lib import *
from sources import train_days, source_fidx, day_of

NP = 8
D = 128
DEEP = np.zeros(N_ET + 1, np.bool_)
for e in (5, 6, 8, 9):
    DEEP[e] = True


@nb.njit(fastmath=True, cache=True)
def _pool(ho, h, skip, hemb, hitem, het, deep, table, X):
    """Pooled unit vectors of events hpos >= skip (recency re-based to hpos - skip). X: (NP, D) zeroed."""
    for j in range(skip, h):
        ei = hemb[ho + j]
        if ei < 0:
            continue
        r = j - skip
        v = np.empty(D, np.float32)
        s = 0.0
        for d in range(D):
            v[d] = np.float32(table[ei, d]); s += v[d] * v[d]
        s = np.sqrt(s) + 1e-6
        for d in range(D):
            v[d] /= s
        e = het[ho + j]
        w = (0.5 ** (r / 1.0), 0.5 ** (r / 3.0), 0.5 ** (r / 10.0), 0.5 ** (r / 40.0))
        for d in range(D):
            X[0, d] += w[0] * v[d]; X[1, d] += w[1] * v[d]; X[2, d] += w[2] * v[d]; X[3, d] += w[3] * v[d]
        if e == 1:
            w5 = 0.5 ** (r / 5.0)
            for d in range(D):
                X[4, d] += w5 * v[d]
        else:
            w10 = 0.5 ** (r / 10.0)
            for d in range(D):
                X[5, d] += w10 * v[d]
        if deep[e]:
            w30 = 0.5 ** (r / 30.0)
            for d in range(D):
                X[6, d] += w30 * v[d]
        it = hitem[ho + j]
        c = 0
        for k in range(skip, h):
            if hitem[ho + k] == it:
                c += 1
        if c >= 2:
            w30 = 0.5 ** (r / 30.0)
            for d in range(D):
                X[7, d] += w30 * c * v[d]
    for q in range(NP):
        s = 0.0
        for d in range(D):
            s += X[q, d] * X[q, d]
        if s > 0:
            s = np.sqrt(s)
            for d in range(D):
                X[q, d] /= s


@nb.njit(parallel=True, fastmath=True, cache=True)
def _train_samples(sel, hist_off, H, hemb, hitem, het, deep, table, Xo, Yo, ok):
    for q in nb.prange(len(sel)):
        f = sel[q]; ho = hist_off[f]; h = H[f]
        tgt = -1
        for j in range(1, min(h, 13)):
            if het[ho + j] == 1 and hemb[ho + j] >= 0:
                tgt = j
                break
        if tgt < 0 or h - tgt - 1 < 3:
            ok[q] = False
            continue
        X = np.zeros((NP, D), np.float32)
        _pool(ho, h, tgt + 1, hemb, hitem, het, deep, table, X)
        for a in range(NP):
            for d in range(D):
                Xo[q, a * D + d] = X[a, d]
        ei = hemb[ho + tgt]
        s = 0.0
        for d in range(D):
            s += np.float32(table[ei, d]) ** 2
        s = np.sqrt(s)
        for d in range(D):
            Yo[q, d] = np.float32(table[ei, d]) / s
        ok[q] = True


@nb.njit(parallel=True, fastmath=True, cache=True)
def _score(cand_off, M, hist_off, H, cemb, hemb, hitem, het, deep, table, Wm, fmodel, out):
    n = len(M)
    for f in nb.prange(n):
        co = cand_off[f]; m = M[f]; ho = hist_off[f]; h = H[f]
        mi = fmodel[f]
        X = np.zeros((NP, D), np.float32)
        _pool(ho, h, 0, hemb, hitem, het, deep, table, X)
        y = np.zeros(D, np.float32)
        nz = False
        for a in range(NP):
            for d in range(D):
                x = X[a, d]
                if x != 0.0:
                    nz = True
                    for k in range(D):
                        y[k] += x * Wm[mi, a * D + d, k]
        yn = 0.0
        for k in range(D):
            yn += y[k] * y[k]
        yn = np.sqrt(yn)
        for i in range(m):
            r = co + i
            ei = cemb[r]
            if ei < 0 or not nz or yn == 0:
                out[r] = np.nan
                continue
            s = 0.0; dt = 0.0
            for k in range(D):
                c = np.float32(table[ei, k]); s += c * c; dt += c * y[k]
            out[r] = dt / (np.sqrt(s) * yn)


def fit(src, n_max=1_500_000, lam=10.0, seed=0):
    f = feeds("train")
    ho = f["hist_off"].to_numpy(); H = f["H"].to_numpy().astype(np.int64)
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(src, min(n_max, len(src)), replace=False))
    table = item_table("train")
    Xo = np.zeros((len(sel), NP * D), np.float32); Yo = np.zeros((len(sel), D), np.float32)
    ok = np.zeros(len(sel), np.bool_)
    _train_samples(sel, ho, H, np.asarray(arr("train", "hist_emb_idx")), np.asarray(arr("train", "hist_item_id")),
                   np.asarray(arr("train", "hist_event_type")).astype(np.int64), DEEP, table, Xo, Yo, ok)
    Xo = Xo[ok]; Yo = Yo[ok]
    Xd = Xo.astype(np.float64)
    A = Xd.T @ Xd + lam * len(Xd) / 1e4 * np.eye(Xd.shape[1])
    B = Xd.T @ Yo.astype(np.float64)
    W = np.linalg.solve(A, B).astype(np.float32)
    pred = Xo @ W
    cos = (pred * Yo).sum(1) / (np.linalg.norm(pred, axis=1) + 1e-9)
    base = (Xo[:, 2 * D:3 * D] * Yo).sum(1)
    log(f"fit: n={len(Xo)} train cos(pred, target) mean {cos.mean():.4f}  vs hl10-pool {base.mean():.4f}")
    return W


def run(split, models):
    f = feeds(split)
    M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
    co = f["cand_off"].to_numpy(); ho = f["hist_off"].to_numpy()
    if split == "train":
        d = day_of(f["timestamp"].to_numpy())
        fm = np.where(d >= models["cut_day"], 0, -1).astype(np.int64)
        Wm = models["early"][None]
    else:
        fm = np.zeros(len(f), np.int64)
        Wm = models["all"][None]
    out = np.full(int(M.sum()), np.nan, np.float32)
    fm2 = np.maximum(fm, 0)
    _score(co, M, ho, H, np.asarray(arr(split, "cand_emb_idx")), np.asarray(arr(split, "hist_emb_idx")),
           np.asarray(arr(split, "hist_item_id")), np.asarray(arr(split, "hist_event_type")).astype(np.int64), DEEP,
           item_table(split), Wm, fm2, out)
    if split == "train":
        rows = np.repeat(fm < 0, M)
        out[rows] = np.nan   # rows before the cutoff day: no strictly-earlier model
    save(split, "seq_nxt_cos", out)


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    sp = sys.argv[1].split(",")
    models = {}
    p = WORK / "nxt_W_all.npy"
    if p.exists():
        models["all"] = np.load(p)
    else:
        models["all"] = fit(source_fidx(None)); np.save(p, models["all"])
    if "train" in sp:
        days = train_days()
        cut = int(days[len(days) // 2])
        p = WORK / f"nxt_W_lt{cut}.npy"
        if p.exists():
            models["early"] = np.load(p)
        else:
            models["early"] = fit(source_fidx(cut)); np.save(p, models["early"])
        models["cut_day"] = cut
    for s in sp:
        run(s, models)
        log(s, "done")
