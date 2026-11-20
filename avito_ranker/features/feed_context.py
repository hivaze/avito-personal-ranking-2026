"""In-feed context: how a candidate stands among the other candidates of its feed.

A feed is a set of up to 300 candidates that compete for the same user. These
20 inputs describe that competition with content embeddings only. Three
families:

Neighbours among the candidates (``ctx_nd_*``, 10 inputs).
    Cosine similarity between candidates of the feed. ``max`` is the
    similarity to the closest other candidate. ``knn_u``, ``knn_h`` and
    ``knn_f1`` are the user similarity, the history similarity and ``f1`` of
    the other candidates, averaged with weights ``exp(20 (cos - 1))``, so only
    near neighbours count; ``knn_u_dev`` and ``knn_h_dev`` are the candidate's
    own value minus that average. ``top5u`` and ``top5h`` are the mean
    similarity to the five candidates that best match the user or the history.
    ``ucent`` and ``hcent`` are the similarity to the user (to the mean of the
    ten newest history items) minus the similarity to the mean candidate.

Competition for a history item, hard (``ctx_hn_*``, 6 inputs).
    For each of the 32 newest history items, rank the feed's candidates by
    similarity to it. ``top1w`` is the recency-weighted share of history items
    for which the candidate is the nearest one, ``rrw`` the weighted mean
    reciprocal rank, ``bestrk`` its rank for the history item it matches best,
    ``margin`` its lead over the best other candidate for that item,
    ``soft05`` a softmax share (temperature 0.05) times the feed size.
    ``e1_rrw`` is ``rrw`` over history events of type 1 only.

Competition for a history item, soft (``ctx_hs_*``, 4 inputs).
    The softmax share again with other settings (``t10``: temperature 0.10;
    ``k128``: 128 history items with slower decay; ``e1``: events of type 1),
    and ``mutual``: the weighted share of history items that are the
    candidate's nearest history item while the candidate is their nearest
    candidate.

Why it is safe. Every value is a function of one feed alone: its candidates'
embeddings, its history, its user embedding and ``f1``. No labels, no
positions, no timestamps, no other feed, no statistics fitted on train. The
same code runs on train, val and test rows.

Run time on 12 threads: 5 minutes for train, 3 seconds each for val and test.
The kernels call BLAS from inside numba threads; run with
``OPENBLAS_NUM_THREADS=1``.
"""
from __future__ import annotations

import numba as nb
import numpy as np

from avito_ranker.data.flatten import (
    feed_layout,
    read_feeds,
    read_flat,
    read_item_embeddings,
)
from avito_ranker.features.store import n_candidates

NEWEST_HISTORY = 32  # history items used for the history similarity of a candidate

NEIGHBOUR_NAMES = (
    "ctx_nd_max", "ctx_nd_top5u", "ctx_nd_top5h", "ctx_nd_knn_u", "ctx_nd_knn_h", "ctx_nd_knn_f1",
    "ctx_nd_ucent", "ctx_nd_hcent", "ctx_nd_knn_u_dev", "ctx_nd_knn_h_dev",
)
HARD_NAMES = ("ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05")
HARD_TYPE1_NAME = "ctx_hn_e1_rrw"
# soft-competition input -> (history items, recency decay, temperature, event type or 0 for all)
SOFT_SETTINGS = {
    "ctx_hs_t10": (32, 0.85, 0.10, 0),
    "ctx_hs_k128": (128, 0.95, 0.05, 0),
    "ctx_hs_e1": (32, 0.85, 0.05, 1),
}
MUTUAL_NAME = "ctx_hs_mutual"  # by-product of the ``ctx_hs_t10`` pass
HARD_ITEMS, HARD_DECAY = 32, 0.85


@nb.njit(parallel=True, fastmath=True, cache=True)
def _base_similarity(cand_off, n_cand, hist_off, n_hist, cand_emb, hist_emb, user_emb, table, out):
    """Per candidate: 0 cosine to the user embedding, 1 max cosine to the 32 newest history items,
    2 cosine to the mean candidate of the feed. NaN where the candidate has no embedding."""
    dim = table.shape[1]
    for f in nb.prange(len(n_cand)):
        u = user_emb[f].astype(np.float32)
        un = np.sqrt((u * u).sum()) + 1e-6
        ho = hist_off[f]
        nk = 0
        newest = np.zeros((NEWEST_HISTORY, dim), np.float32)
        for j in range(n_hist[f]):
            if nk >= NEWEST_HISTORY:
                break
            ei = hist_emb[ho + j]
            if ei < 0:
                continue
            v = table[ei].astype(np.float32)
            vn = np.sqrt((v * v).sum()) + 1e-6
            v = v / vn
            newest[nk] = v
            nk += 1
        co = cand_off[f]
        m = n_cand[f]
        mean = np.zeros(dim, np.float32)
        unit = np.zeros((m, dim), np.float32)
        for i in range(m):
            ei = cand_emb[co + i]
            if ei < 0:
                out[co + i, 0] = np.nan
                out[co + i, 1] = np.nan
                continue
            v = table[ei].astype(np.float32)
            vn = np.sqrt((v * v).sum()) + 1e-6
            dot = (v * u).sum()
            out[co + i, 0] = dot / (vn * un)
            v = v / vn
            unit[i] = v
            mean += v
            if nk > 0:
                mx = -2.0
                for k in range(nk):
                    s = (v * newest[k]).sum()
                    if s > mx:
                        mx = s
                out[co + i, 1] = mx
            else:
                out[co + i, 1] = np.nan
        mean_norm = np.sqrt((mean * mean).sum())
        for i in range(m):
            if cand_emb[co + i] >= 0 and mean_norm > 0:
                out[co + i, 2] = (unit[i] * mean).sum() / mean_norm
            else:
                out[co + i, 2] = np.nan


def base_similarity(split: str) -> dict[str, np.ndarray]:
    """``cos_user``, ``max_history`` and ``centrality`` of every candidate of a split (float32).

    The first two feed the neighbour kernel; ``centrality`` is one of the
    fine-tune extras.
    """
    cand_off, n_cand, hist_off, n_hist = feed_layout(read_feeds(split))
    out = np.empty((n_candidates(split), 3), np.float32)
    _base_similarity(
        cand_off, n_cand, hist_off, n_hist,
        np.asarray(read_flat(split, "cand_emb_idx")), np.asarray(read_flat(split, "hist_emb_idx")),
        np.asarray(read_flat(split, "user_emb")), np.ascontiguousarray(read_item_embeddings(split)), out,
    )
    return {"cos_user": out[:, 0], "max_history": out[:, 1], "centrality": out[:, 2]}


@nb.njit(parallel=True, fastmath=True, cache=True)
def _neighbours(cand_off, n_cand, hist_off, n_hist, cand_emb, hist_emb, table, user_emb, cos_user, max_history, f1,
                out):
    """Columns of ``out`` in the order of ``NEIGHBOUR_NAMES``.

    Names inside the kernel, per feed ``f`` and candidate ``i``:

    ``V``, ``ok``     unit content embedding of each candidate, and whether it has one
    ``S``             cosine between every two candidates (``V @ V.T``)
    ``fm``            unit mean of the candidates' embeddings (the feed centre)
    ``u``, ``uok``    unit user embedding, and whether it is non-zero
    ``h``, ``hok``    unit mean of the 10 newest history embeddings, and whether there is one
    ``cu``, ``ch``    each candidate's cosine to the user and best cosine to the history, -9 when missing;
                      ``by_user`` and ``by_history`` order the candidates by them
    ``w``             weight of neighbour ``j``: ``exp(20 (S[i, j] - 1))``
    ``wsu``, ``wu``   weighted sum of the neighbours' cosine to the user, and the sum of its weights
    ``wsh``, ``wh``   the same for the neighbours' best cosine to the history
    ``wsf``, ``wf``   the same for the neighbours' ``f1``
    ``mx``, ``no``    largest cosine to another candidate, and the number of candidates compared
    ``st``, ``t``     sum and count of the cosines to the 5 candidates closest to the user (or history)
    ``cfm``           the candidate's cosine to the feed centre
    """
    dim = table.shape[1]
    for f in nb.prange(len(n_cand)):
        co = cand_off[f]
        m = n_cand[f]
        V = np.zeros((m, dim), np.float32)
        ok = np.zeros(m, np.bool_)
        fm = np.zeros(dim, np.float32)
        for i in range(m):
            ei = cand_emb[co + i]
            if ei < 0:
                continue
            v = table[ei].astype(np.float32)
            nv = np.sqrt((v * v).sum()) + 1e-6
            V[i] = v / nv
            ok[i] = True
            fm += V[i]
        fmn = np.sqrt((fm * fm).sum())
        if fmn > 0:
            fm = fm / fmn
        u = user_emb[f].astype(np.float32)
        un = np.sqrt((u * u).sum())
        uok = un > 0
        if uok:
            u = u / un
        h = np.zeros(dim, np.float32)
        ho = hist_off[f]
        nh = 0
        for j in range(n_hist[f]):
            ei = hist_emb[ho + j]
            if ei < 0:
                continue
            v = table[ei].astype(np.float32)
            nv = np.sqrt((v * v).sum()) + 1e-6
            h += v / nv
            nh += 1
            if nh >= 10:
                break
        hn = np.sqrt((h * h).sum())
        hok = hn > 0
        if hok:
            h = h / hn
        S = V @ V.T
        # A zero user vector gives cosine 0 (not NaN) for every candidate; such a feed has no
        # meaningful "closest to the user" order, so its user-based values are left missing.
        cu = np.empty(m)
        ch = np.empty(m)
        for i in range(m):
            cu[i] = cos_user[co + i] if (ok[i] and uok and not np.isnan(cos_user[co + i])) else -9.0
            ch[i] = max_history[co + i] if (ok[i] and not np.isnan(max_history[co + i])) else -9.0
        by_user = np.argsort(-cu)
        by_history = np.argsort(-ch)
        for i in range(m):
            if not ok[i]:
                for k in range(10):
                    out[co + i, k] = np.nan
                continue
            mx = -2.0
            no = 0
            wsu = 0.0
            wsh = 0.0
            wsf = 0.0
            wu = 0.0
            wh = 0.0
            wf = 0.0
            for j in range(m):
                if j == i or not ok[j]:
                    continue
                s = S[i, j]
                no += 1
                if s > mx:
                    mx = s
                w = np.exp(20.0 * (s - 1.0))
                if uok and not np.isnan(cos_user[co + j]):
                    wsu += w * cos_user[co + j]
                    wu += w
                if not np.isnan(max_history[co + j]):
                    wsh += w * max_history[co + j]
                    wh += w
                if not np.isnan(f1[co + j]):
                    wsf += w * f1[co + j]
                    wf += w
            out[co + i, 0] = mx if no > 0 else np.nan
            t = 0
            st = 0.0
            for r in range(m):
                j = by_user[r]
                if j == i or cu[j] < -5:
                    continue
                st += S[i, j]
                t += 1
                if t >= 5:
                    break
            out[co + i, 1] = st / t if t > 0 else np.nan
            t = 0
            st = 0.0
            for r in range(m):
                j = by_history[r]
                if j == i or ch[j] < -5:
                    continue
                st += S[i, j]
                t += 1
                if t >= 5:
                    break
            out[co + i, 2] = st / t if t > 0 else np.nan
            out[co + i, 3] = wsu / wu if wu > 0 else np.nan
            out[co + i, 4] = wsh / wh if wh > 0 else np.nan
            out[co + i, 5] = wsf / wf if wf > 0 else np.nan
            cfm = (V[i] * fm).sum()
            out[co + i, 6] = (V[i] * u).sum() - cfm if uok else np.nan
            out[co + i, 7] = (V[i] * h).sum() - cfm if hok else np.nan
            out[co + i, 8] = cos_user[co + i] - out[co + i, 3]
            out[co + i, 9] = max_history[co + i] - out[co + i, 4]


@nb.njit(parallel=True, fastmath=True, cache=True)
def _hard_competition(cand_off, n_cand, hist_off, n_hist, cand_emb, hist_emb, hist_mask, table, n_items, decay, out):
    """Columns of ``out`` in the order of ``HARD_NAMES``."""
    dim = table.shape[1]
    for f in nb.prange(len(n_cand)):
        co = cand_off[f]
        m = n_cand[f]
        Hm = np.zeros((n_items, dim), np.float32)
        nk = 0
        ho = hist_off[f]
        for j in range(n_hist[f]):
            ei = hist_emb[ho + j]
            if ei < 0 or not hist_mask[ho + j]:
                continue
            v = table[ei].astype(np.float32)
            Hm[nk] = v / (np.sqrt((v * v).sum()) + 1e-6)
            nk += 1
            if nk >= n_items:
                break
        V = np.zeros((m, dim), np.float32)
        ok = np.zeros(m, np.bool_)
        for i in range(m):
            ei = cand_emb[co + i]
            if ei >= 0:
                v = table[ei].astype(np.float32)
                V[i] = v / (np.sqrt((v * v).sum()) + 1e-6)
                ok[i] = True
        if nk == 0:
            for i in range(m):
                for c in range(5):
                    out[co + i, c] = np.nan
            continue
        S = V @ Hm[:nk].T
        for i in range(m):
            if not ok[i]:
                for j in range(nk):
                    S[i, j] = -9.0
        top1w = np.zeros(m)
        rrw = np.zeros(m)
        s05 = np.zeros(m)
        wsum = 0.0
        R = np.zeros((m, nk), np.int32)  # rank of every candidate for every history item
        best_other = np.zeros((m, nk), np.float32)  # best similarity among the other candidates
        for j in range(nk):
            w = decay ** j
            wsum += w
            col = S[:, j].copy()
            o = np.argsort(-col)
            for r in range(m):
                R[o[r], j] = r
            b1 = col[o[0]]
            b2 = col[o[1]] if m > 1 else -9.0
            for i in range(m):
                best_other[i, j] = b2 if i == o[0] else b1
            top1w[o[0]] += w
            for i in range(m):
                rrw[i] += w / (1.0 + R[i, j])
            z05 = 0.0
            for i in range(m):
                if ok[i]:
                    z05 += np.exp((col[i] - b1) / 0.05)
            for i in range(m):
                if ok[i]:
                    s05[i] += w * np.exp((col[i] - b1) / 0.05) / z05
        for i in range(m):
            if not ok[i]:
                for c in range(5):
                    out[co + i, c] = np.nan
                continue
            jb = 0
            for j in range(1, nk):
                if S[i, j] > S[i, jb]:
                    jb = j
            out[co + i, 0] = top1w[i] / wsum
            out[co + i, 1] = rrw[i] / wsum
            out[co + i, 2] = R[i, jb] / (m - 1) if m > 1 else 0.0
            out[co + i, 3] = S[i, jb] - best_other[i, jb]
            out[co + i, 4] = s05[i] / wsum * m


@nb.njit(parallel=True, fastmath=True, cache=True)
def _soft_competition(cand_off, n_cand, hist_off, n_hist, cand_emb, hist_emb, hist_mask, table, n_items, decay, tau,
                      out_mass, out_mutual):
    """Per candidate: the softmax share it wins over the newest history items (``out_mass``, times the feed
    size) and the weighted share of history items with which it is mutually nearest (``out_mutual``)."""
    dim = table.shape[1]
    for f in nb.prange(len(n_cand)):
        co = cand_off[f]
        m = n_cand[f]
        Hm = np.zeros((n_items, dim), np.float32)
        nk = 0
        ho = hist_off[f]
        for j in range(n_hist[f]):
            ei = hist_emb[ho + j]
            if ei < 0 or not hist_mask[ho + j]:
                continue
            v = table[ei].astype(np.float32)
            Hm[nk] = v / (np.sqrt((v * v).sum()) + 1e-6)
            nk += 1
            if nk >= n_items:
                break
        V = np.zeros((m, dim), np.float32)
        ok = np.zeros(m, np.bool_)
        for i in range(m):
            ei = cand_emb[co + i]
            if ei >= 0:
                v = table[ei].astype(np.float32)
                V[i] = v / (np.sqrt((v * v).sum()) + 1e-6)
                ok[i] = True
        if nk == 0:
            for i in range(m):
                out_mass[co + i] = np.nan
                out_mutual[co + i] = np.nan
            continue
        S = V @ Hm[:nk].T
        mass = np.zeros(m)
        mut = np.zeros(m)
        nearest_history = np.zeros(m, np.int64)
        for i in range(m):
            b = 0
            for j in range(1, nk):
                if S[i, j] > S[i, b]:
                    b = j
            nearest_history[i] = b
        wsum = 0.0
        for j in range(nk):
            w = decay ** j
            wsum += w
            b1 = -9.0
            ib = -1
            for i in range(m):
                if ok[i] and S[i, j] > b1:
                    b1 = S[i, j]
                    ib = i
            if ib < 0:
                continue
            z = 0.0
            for i in range(m):
                if ok[i]:
                    z += np.exp((S[i, j] - b1) / tau)
            for i in range(m):
                if ok[i]:
                    mass[i] += w * np.exp((S[i, j] - b1) / tau) / z
            if nearest_history[ib] == j:
                mut[ib] += w
        for i in range(m):
            if ok[i]:
                out_mass[co + i] = mass[i] / wsum * m
                out_mutual[co + i] = mut[i] / wsum
            else:
                out_mass[co + i] = np.nan
                out_mutual[co + i] = np.nan


def build_neighbours(split: str, base: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """The ten ``ctx_nd_*`` inputs of a split. ``base`` may pass the result of ``base_similarity``."""
    base = base or base_similarity(split)
    cand_off, n_cand, hist_off, n_hist = feed_layout(read_feeds(split))
    out = np.empty((n_candidates(split), len(NEIGHBOUR_NAMES)), np.float32)
    _neighbours(
        cand_off, n_cand, hist_off, n_hist,
        read_flat(split, "cand_emb_idx", mmap=False).astype(np.int64),
        read_flat(split, "hist_emb_idx", mmap=False).astype(np.int64),
        np.ascontiguousarray(read_item_embeddings(split)), read_flat(split, "user_emb", mmap=False),
        base["cos_user"].astype(np.float32), base["max_history"].astype(np.float32),
        read_flat(split, "cand_f1", mmap=False).astype(np.float32), out,
    )
    return {name: out[:, i] for i, name in enumerate(NEIGHBOUR_NAMES)}


def build_history_competition(split: str) -> dict[str, np.ndarray]:
    """The six ``ctx_hn_*`` and four ``ctx_hs_*`` inputs of a split."""
    cand_off, n_cand, hist_off, n_hist = feed_layout(read_feeds(split))
    table = np.ascontiguousarray(read_item_embeddings(split))
    cand_emb = read_flat(split, "cand_emb_idx", mmap=False).astype(np.int64)
    hist_emb = read_flat(split, "hist_emb_idx", mmap=False).astype(np.int64)
    event_type = read_flat(split, "hist_event_type", mmap=False)
    every_event = np.ones(len(hist_emb), np.bool_)
    columns = {}

    def hard(mask: np.ndarray) -> np.ndarray:
        """The five hard-competition columns over the history events selected by ``mask``."""
        out = np.empty((len(cand_emb), len(HARD_NAMES)), np.float32)
        _hard_competition(cand_off, n_cand, hist_off, n_hist, cand_emb, hist_emb, mask, table,
                          HARD_ITEMS, HARD_DECAY, out)
        return out

    out = hard(every_event)
    columns.update({name: out[:, i] for i, name in enumerate(HARD_NAMES)})
    columns[HARD_TYPE1_NAME] = hard(event_type == 1)[:, HARD_NAMES.index("ctx_hn_rrw")]

    for name, (n_items, decay, tau, only_type) in SOFT_SETTINGS.items():
        mask = every_event if only_type == 0 else (event_type == only_type)
        mass = np.empty(len(cand_emb), np.float32)
        mutual = np.empty(len(cand_emb), np.float32)
        _soft_competition(cand_off, n_cand, hist_off, n_hist, cand_emb, hist_emb, mask, table,
                          n_items, decay, tau, mass, mutual)
        columns[name] = mass
        if name == "ctx_hs_t10":
            columns[MUTUAL_NAME] = mutual
    return columns


def build_feed_context(split: str) -> dict[str, np.ndarray]:
    """All 20 in-feed context inputs of a split, keyed by file name."""
    return {**build_neighbours(split), **build_history_competition(split)}
