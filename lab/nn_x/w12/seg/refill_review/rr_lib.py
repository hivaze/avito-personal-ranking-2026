"""Reviewer's own library for the refill-replay review (label-free part). Nothing of nn_x/w12/seg/refill/*.py is imported.
Aligner: patience anchors (keys unique in both sequences, longest increasing subsequence) + multiset pairing inside gaps.
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_k, "12")
import bisect
from collections import Counter, defaultdict

import numpy as np
import polars as pl

ROOT = "<repo>"
FLAT = f"{ROOT}/data/flat"
HERE = f"{ROOT}/nn_x/w12/seg/refill_review"
SPLITS = ("train", "val", "test")
BOUND = 1790497920


class Tables:
    def __init__(self):
        self.uid, self.ts, self.H, self.hoff, self.M, self.coff, self.n = {}, {}, {}, {}, {}, {}, {}
        self.item, self.et, self.es, self.hf = {}, {}, {}, {}
        for s in SPLITS:
            f = pl.read_parquet(f"{FLAT}/{s}/feeds.parquet")
            self.uid[s] = f["user_id"].to_numpy().astype(np.int64)
            self.ts[s] = f["timestamp"].to_numpy().astype(np.int64)
            self.H[s] = f["H"].to_numpy().astype(np.int64)
            self.hoff[s] = f["hist_off"].to_numpy().astype(np.int64)
            self.M[s] = f["M"].to_numpy().astype(np.int64)
            self.coff[s] = f["cand_off"].to_numpy().astype(np.int64)
            self.n[s] = len(f)
            self.item[s] = np.load(f"{FLAT}/{s}/hist_item_id.npy", mmap_mode="r")
            self.et[s] = np.load(f"{FLAT}/{s}/hist_event_type.npy", mmap_mode="r")
            self.es[s] = np.load(f"{FLAT}/{s}/hist_event_source.npy", mmap_mode="r")
            self.hf[s] = np.load(f"{FLAT}/{s}/hist_fidx.npy", mmap_mode="r")
        # per-user index over all splits
        u = np.concatenate([self.uid[s] for s in SPLITS])
        t = np.concatenate([self.ts[s] for s in SPLITS])
        sc = np.concatenate([np.full(self.n[s], i, np.int64) for i, s in enumerate(SPLITS)])
        fx = np.concatenate([np.arange(self.n[s]) for s in SPLITS])
        o = np.argsort(u, kind="stable")
        self._u, self._t, self._sc, self._fx = u[o], t[o], sc[o], fx[o]

    def holdout(self):
        pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
        assert (pr["fidx"] == np.arange(self.n["train"])).all()
        p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
        p[self.M["train"] < 50] = 0.0
        cand = np.where((p > 0.9) & (self.M["train"] >= 50))[0]
        return np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False)), p

    def earlier(self, s, f, splits=(0, 1, 2)):
        """feeds of the same user with a STRICTLY smaller timestamp, most recent first: [(split code, feed)]."""
        u, t = self.uid[s][f], self.ts[s][f]
        if u == 0:
            return []
        lo, hi = np.searchsorted(self._u, u, "left"), np.searchsorted(self._u, u, "right")
        tt, cc, ff = self._t[lo:hi], self._sc[lo:hi], self._fx[lo:hi]
        m = (tt < t) & np.isin(cc, splits)
        o = np.argsort(-tt[m], kind="stable")
        return [(int(c), int(g), int(x)) for c, g, x in zip(cc[m][o], ff[m][o], tt[m][o])]

    def hist(self, sc, f, cap=None):
        s = SPLITS[sc]
        o, h = int(self.hoff[s][f]), int(self.H[s][f])
        if cap is not None:
            h = min(h, cap)
        it = np.asarray(self.item[s][o:o + h]).astype(np.int64)
        et = np.asarray(self.et[s][o:o + h]).astype(np.int64)
        es = np.asarray(self.es[s][o:o + h]).astype(np.int64)
        return it * 100000 + et * 1000 + es, np.arange(o, o + h, dtype=np.int64), it

    def qualifying(self, s, ids):
        ids = np.asarray(ids)
        c = ids[(self.H[s][ids] == 256) & (self.uid[s][ids] != 0)]
        # strictly earlier snapshot exists: min timestamp of the user over all splits < this timestamp
        o = np.argsort(self._u, kind="stable")  # already sorted; compute per-user min ts
        ub, st = np.unique(self._u, return_index=True)
        mn = np.minimum.reduceat(self._t, st)
        tmin = mn[np.searchsorted(ub, self.uid[s][c])]
        return c[tmin < self.ts[s][c]]


def _lis(seq):
    """indices of a longest strictly increasing subsequence of seq."""
    tails, tails_i, prev = [], [], [-1] * len(seq)
    for i, v in enumerate(seq):
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
            tails_i.append(i)
        else:
            tails[k] = v
            tails_i[k] = i
        prev[i] = tails_i[k - 1] if k > 0 else -1
    out, i = [], tails_i[-1] if tails_i else -1
    while i >= 0:
        out.append(i)
        i = prev[i]
    return out[::-1]


def pair(L, E):
    """L, E int64 key arrays (newest first). Returns mL [len L] (index in E or -1), mE [len E] (index in L or -1), anchors [(i, j)]."""
    cl, ce = Counter(L.tolist()), Counter(E.tolist())
    posE = {k: j for j, k in enumerate(E.tolist()) if ce[k] == 1}
    cand = [(i, posE[k]) for i, k in enumerate(L.tolist()) if cl[k] == 1 and k in posE]
    keep = _lis([j for _, j in cand])
    anc = [cand[k] for k in keep]
    mL, mE = np.full(len(L), -1, np.int64), np.full(len(E), -1, np.int64)
    for i, j in anc:
        mL[i], mE[j] = j, i
    bounds = [(-1, -1)] + anc + [(len(L), len(E))]
    for (i0, j0), (i1, j1) in zip(bounds[:-1], bounds[1:]):
        if i1 - i0 <= 1 or j1 - j0 <= 1:
            continue
        slots = defaultdict(list)
        for j in range(j0 + 1, j1):
            slots[int(E[j])].append(j)
        for i in range(i0 + 1, i1):
            q = slots.get(int(L[i]))
            if q:
                j = q.pop(0)
                mL[i], mE[j] = j, i
    return mL, mE, anc


def run_len(mL):
    """longest stretch of consecutive L positions paired with consecutive E positions."""
    best = cur = 0
    prev = None
    for i in range(len(mL)):
        j = mL[i]
        if j >= 0 and prev is not None and j == prev + 1:
            cur += 1
        elif j >= 0:
            cur = 1
        else:
            cur = 0
        prev = j if j >= 0 else None
        best = max(best, cur)
    return best


def timeline(T, s, f, cap=256, cap_snap=None, snaps=None, splits=(0, 1, 2)):
    """Union timeline of feed f of split s with its strictly earlier snapshots (my aligner).
    Returns dict: key, sc, row, lpos (stored index or -1), item; stats."""
    cap_snap = cap if cap_snap is None else cap_snap
    sc0 = SPLITS.index(s)
    k0, r0, i0 = T.hist(sc0, f, cap)
    n0 = len(k0)
    U = dict(key=k0.copy(), sc=np.full(n0, sc0, np.int64), row=r0.copy(), lpos=np.arange(n0, dtype=np.int64), item=i0.copy(),
             src=np.full(n0, -1, np.int64), spos=np.arange(n0, dtype=np.int64))
    snaps = T.earlier(s, f, splits) if snaps is None else snaps
    st = dict(n_snap=len(snaps), n_used=0, deep=-1, n_anchor1=0)
    paired_stored = np.zeros(n0, bool)
    for g, (sc, fe, _t) in enumerate(snaps):
        ek, er, ei = T.hist(sc, fe, cap_snap)
        if len(ek) < 16:
            continue
        mL, mE, anc = pair(U["key"], ek)
        if len(anc) < 16 or run_len(mL) < 8:
            continue
        if st["n_used"] == 0:
            st["n_anchor1"] = len(anc)
        st["n_used"] += 1
        lp = U["lpos"][mL >= 0]
        paired_stored[lp[lp >= 0]] = True
        new = np.where(mE < 0)[0]
        if len(new) == 0:
            continue
        # place every old-only event after the timeline's events of its gap: position = index of the next ANCHOR in U (or end)
        aj = np.array([j for _, j in anc])
        ai = np.array([i for i, _ in anc])
        nxt = np.searchsorted(aj, new)  # anchors before the event in E
        pos = np.where(nxt < len(ai), ai[np.minimum(nxt, len(ai) - 1)] - 0.5, float(len(U["key"])))
        allpos = np.r_[np.arange(len(U["key"]), dtype=np.float64), pos]
        sec = np.r_[np.zeros(len(U["key"])), new.astype(np.float64) + 1]
        o = np.lexsort((sec, allpos))
        k = len(new)
        U = dict(key=np.r_[U["key"], ek[new]][o], sc=np.r_[U["sc"], np.full(k, sc)][o], row=np.r_[U["row"], er[new]][o],
                 lpos=np.r_[U["lpos"], np.full(k, -1)][o], item=np.r_[U["item"], ei[new]][o],
                 src=np.r_[U["src"], np.full(k, g)][o], spos=np.r_[U["spos"], new][o])
    st["deep"] = int(np.where(paired_stored)[0].max()) if paired_stored.any() else -1
    last = int(np.where(U["lpos"] == n0 - 1)[0][0])
    old = U["lpos"] < 0
    st["last"] = last
    st["n_T"] = int(old[last + 1:].sum())
    st["n_U"] = int(old.sum())
    return U, st


def cstat(x, clu):
    """mean, cluster-robust SE (clusters = users)."""
    x = np.asarray(x, np.float64)
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    mu = x.mean()
    _, inv = np.unique(clu, return_inverse=True)
    r = np.bincount(inv, weights=x - mu)
    g = len(r)
    return float(mu), float(np.sqrt((r ** 2).sum() * g / max(g - 1, 1)) / n)


def boot_ratio(num, den, clu, nb=4000, seed=2026):
    _, inv = np.unique(clu, return_inverse=True)
    g = inv.max() + 1
    sn, sd = np.bincount(inv, weights=num, minlength=g), np.bincount(inv, weights=den, minlength=g)
    rng = np.random.default_rng(seed)
    out = np.empty(nb)
    for b in range(nb):
        ix = rng.integers(0, g, g)
        out[b] = sn[ix].sum() / sd[ix].sum()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def my_ndcg(scores, y, M):
    """per-feed NDCG, binary gain, log2 discount, pessimistic ties (negatives first among equal scores), 1 if no positive."""
    out = np.ones(len(M))
    o = 0
    for i, m in enumerate(M):
        s, t = scores[o:o + m], y[o:o + m]
        o += m
        k = int(t.sum())
        if k == 0:
            continue
        order = np.lexsort((t, -s))
        disc = 1.0 / np.log2(np.arange(m) + 2.0)
        out[i] = (t[order] * disc).sum() / disc[:k].sum()
    return out
