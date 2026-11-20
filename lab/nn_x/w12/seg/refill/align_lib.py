"""Refill replay, step 1: alignment of a stored history with the same user's STRICTLY EARLIER snapshots.
Label-free. Reads data/flat/* memory-mapped (read-only). No project module is imported.

Stored order (readme of the organisers, data.py docstring, verified in s1_align.py): index 0 = newest event.
Event key = (item_id, event_type, event_source). Histories are not a shifted log (per-type caps, unstable order of
simultaneous events), so two snapshots are aligned by a longest common subsequence (LCS) of their keys and merged into one
union timeline U (newest first):
  anchors          = LCS pairs (monotone);
  swap repair      = an unmatched event of the earlier snapshot whose key is unmatched in U within +-W anchors is the same
                     event in another tie order: paired, not inserted;
  gap convention   = between two consecutive anchors the events U already holds come first, then the events only the earlier
                     snapshot holds (in its own order); after the last anchor likewise.
Snapshots are merged from the most recent to the oldest. A snapshot is merged only if its LCS with U has >= MIN_ANCH anchors
and a contiguous run (consecutive in both sequences) of >= MIN_RUN.
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_k, "12")
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
HERE = ROOT / "nn_x/w12/seg/refill"
OUT = HERE / "out"
SPL = ["train", "val", "test"]
MIN_ANCH, MIN_RUN, W_SWAP = 16, 8, 3
BOUND = 1790497920  # public-like / private-like test boundary used by the project (review/z_sizing.py)


class Flat:
    def __init__(self):
        self.uid, self.ts, self.H, self.hoff, self.M, self.coff = {}, {}, {}, {}, {}, {}
        self.item, self.et, self.es = {}, {}, {}
        for s in SPL:
            f = pl.read_parquet(FLAT / s / "feeds.parquet")
            self.uid[s] = f["user_id"].to_numpy().astype(np.int64)
            self.ts[s] = f["timestamp"].to_numpy().astype(np.int64)
            self.H[s] = f["H"].to_numpy().astype(np.int64)
            self.hoff[s] = f["hist_off"].to_numpy().astype(np.int64)
            self.M[s] = f["M"].to_numpy().astype(np.int64)
            self.coff[s] = f["cand_off"].to_numpy().astype(np.int64)
            self.item[s] = np.load(FLAT / s / "hist_item_id.npy", mmap_mode="r")
            self.et[s] = np.load(FLAT / s / "hist_event_type.npy", mmap_mode="r")
            self.es[s] = np.load(FLAT / s / "hist_event_source.npy", mmap_mode="r")
        # user index over all splits: sorted by (uid, ts)
        u = np.concatenate([self.uid[s] for s in SPL])
        t = np.concatenate([self.ts[s] for s in SPL])
        sp = np.concatenate([np.full(len(self.uid[s]), i, np.int8) for i, s in enumerate(SPL)])
        fx = np.concatenate([np.arange(len(self.uid[s])) for s in SPL])
        o = np.lexsort((t, u))
        self.iu, self.it, self.isp, self.ifx = u[o], t[o], sp[o], fx[o]

    def holdout(self):
        """the fixed 30,000 holdout feeds, exactly as train.py --holdout 30000 / holdout_eval.py."""
        pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
        pr[self.M["train"] < 50] = 0.0
        cand = np.where((pr > 0.9) & (self.M["train"] >= 50))[0]
        return np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False)), pr

    def earlier(self, split, f):
        """strictly earlier snapshots of the same user (any split), most recent first: list of (split_code, fidx)."""
        u, t = self.uid[split][f], self.ts[split][f]
        if u == 0:
            return []
        lo, hi = np.searchsorted(self.iu, u), np.searchsorted(self.iu, u, side="right")
        m = self.it[lo:hi] < t
        idx = np.arange(lo, hi)[m][::-1]
        return [(int(self.isp[i]), int(self.ifx[i])) for i in idx]

    def has_any_other(self, split, f):
        u = self.uid[split][f]
        lo, hi = np.searchsorted(self.iu, u), np.searchsorted(self.iu, u, side="right")
        return hi - lo > 1

    def seq(self, sc, f, cap=None):
        """keys [h] int64 and global history rows [h] of feed f of split code sc (newest first), cut to the newest `cap`."""
        s = SPL[sc]
        o, h = self.hoff[s][f], self.H[s][f]
        if cap is not None:
            h = min(h, cap)
        it = np.asarray(self.item[s][o:o + h]).astype(np.int64)
        key = it * 4096 + np.asarray(self.et[s][o:o + h]).astype(np.int64) * 64 + np.asarray(self.es[s][o:o + h]).astype(np.int64)
        return key, np.arange(o, o + h, dtype=np.int64)


@nb.njit(cache=False)
def lcs(a, b):
    """LCS of int64 arrays a, b; returns matched index arrays (earliest-possible matches on both sides)."""
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), np.int32)
    for i in range(n - 1, -1, -1):
        ai = a[i]
        for j in range(m - 1, -1, -1):
            if ai == b[j]:
                dp[i, j] = dp[i + 1, j + 1] + 1
            else:
                x, y = dp[i + 1, j], dp[i, j + 1]
                dp[i, j] = x if x >= y else y
    L = dp[0, 0]
    ia, ib = np.empty(L, np.int64), np.empty(L, np.int64)
    i = j = k = 0
    while i < n and j < m and k < L:
        if a[i] == b[j] and dp[i, j] == dp[i + 1, j + 1] + 1:
            ia[k], ib[k] = i, j
            k += 1
            i += 1
            j += 1
        elif dp[i + 1, j] >= dp[i, j + 1]:
            i += 1
        else:
            j += 1
    return ia, ib


def longest_run(ia, ib):
    if len(ia) == 0:
        return 0
    brk = np.r_[True, (np.diff(ia) != 1) | (np.diff(ib) != 1)]
    g = np.cumsum(brk)
    return int(np.bincount(g).max())


def merge(U, Ek, Erow, Esc, gen):
    """U: dict(key, sc, row, lpos, gen) arrays of the union timeline; (Ek, Erow) one earlier snapshot of split code Esc.
    Returns (U', info) ; U unchanged when the snapshot fails the anchor criterion."""
    ia, ib = lcs(U["key"], Ek)
    na, run = len(ia), longest_run(ia, ib)
    info = dict(n_anchor=na, run=run, merged=False, n_new=0, n_swap=0)
    if na < MIN_ANCH or run < MIN_RUN:
        return U, info
    nU, nE = len(U["key"]), len(Ek)
    um = np.zeros(nU, bool)
    um[ia] = True
    em = np.zeros(nE, bool)
    em[ib] = True
    uq = np.where(~um)[0]
    ukeys = U["key"][uq]
    used = np.zeros(len(uq), bool)
    ins_pos, ins_q = [], []
    for q in np.where(~em)[0]:
        p = int(np.searchsorted(ib, q))  # anchors before q
        lo = ia[p - 1 - W_SWAP] if p - 1 - W_SWAP >= 0 else max(int(ia[0]) - W_SWAP, 0)
        hi = ia[p + W_SWAP] if p + W_SWAP < na else min(int(ia[-1]) + W_SWAP, nU - 1)
        c = np.where((ukeys == Ek[q]) & ~used & (uq >= lo) & (uq <= hi))[0]
        if len(c):  # same event in another tie order
            ref = ia[p] if p < na else nU
            used[c[np.argmin(np.abs(uq[c] - ref))]] = True
            info["n_swap"] += 1
            continue
        ins_pos.append(float(ia[p]) - 0.5 if p < na else float(nU))
        ins_q.append(q)
    if not ins_q:
        info["merged"] = True
        info["anch_u"] = ia
        return U, info
    ins_q = np.array(ins_q)
    pos = np.r_[np.arange(nU, dtype=np.float64), np.array(ins_pos)]
    sec = np.r_[np.zeros(nU), ins_q.astype(np.float64)]
    o = np.lexsort((sec, pos))
    k = len(ins_q)
    V = dict(key=np.r_[U["key"], Ek[ins_q]][o], sc=np.r_[U["sc"], np.full(k, Esc, np.int8)][o],
             row=np.r_[U["row"], Erow[ins_q]][o], lpos=np.r_[U["lpos"], np.full(k, -1, np.int16)][o],
             gen=np.r_[U["gen"], np.full(k, gen, np.int16)][o])
    info.update(merged=True, n_new=k, anch_u=ia)
    return V, info


def union_timeline(F, split, f, cap=256, snaps=None, cap_snap=None):
    """Union timeline of feed f (stored history cut to the newest `cap`) with all strictly earlier snapshots (each cut to
    its newest `cap_snap`, default `cap`: with cap < 256 this is the 'pretend the cap is `cap`' world of the positive check).
    Returns U and a dict of alignment statistics."""
    cap_snap = cap if cap_snap is None else cap_snap
    sc0 = SPL.index(split)
    k0, r0 = F.seq(sc0, f, cap)
    n0 = len(k0)
    U = dict(key=k0, sc=np.full(n0, sc0, np.int8), row=r0, lpos=np.arange(n0, dtype=np.int16), gen=np.zeros(n0, np.int16))
    snaps = F.earlier(split, f) if snaps is None else snaps
    st = dict(n_snap=len(snaps), n_merged=0, n_swap=0, first_anchor=0, first_run=0, deep_anchor=-1)
    anchored = np.zeros(n0, bool)
    for g, (sc, fe) in enumerate(snaps, 1):
        ek, er = F.seq(sc, fe, cap_snap)
        if len(ek) == 0:
            continue
        lp_before = U["lpos"]
        U, info = merge(U, ek, er, sc, g)
        if info["merged"]:
            if st["n_merged"] == 0:
                st["first_anchor"], st["first_run"] = info["n_anchor"], info["run"]
            st["n_merged"] += 1
            st["n_swap"] += info["n_swap"]
            a = lp_before[info["anch_u"]]
            anchored[a[a >= 0]] = True
    st["deep_anchor"] = int(np.where(anchored)[0].max()) if anchored.any() else -1
    st["n_anch_L"] = int(anchored.sum())
    iL = int(np.where(U["lpos"] == n0 - 1)[0][0])
    old = U["lpos"] < 0
    st["n_T"] = int(old[iL + 1:].sum())      # events older than the last stored event
    st["n_U"] = int(old.sum())               # all events the earlier snapshots hold and the stored history lacks
    st["iL"] = iL
    return U, st
