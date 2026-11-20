"""Within-feed transforms of a candidate column: rank, z-score, gap to the best.

A ranking model compares candidates of one feed, so the position of a value
inside its feed often says more than the value. The stored inputs were made
with two slightly different definitions, and both must be reproduced exactly:

``avito_ranker.ensemble.zscore.feed_rank_z``
    Ordinal rank (ties broken by position) scaled to [0, 1], and a z-score
    with ``std + 1e-9`` in the denominator. No missing-value handling. Used
    for the encoder scores and for the leave-user-out counts. It is the
    kernel of the ensemble and lives there.
``rank_zscore_gap`` (here)
    Average rank for ties scaled to [0, 1], a z-score over the non-missing
    values with ``std + 1e-6``, and the gap to the feed maximum. Used for the
    co-visitation rank.
"""
from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(parallel=True, cache=True)
def rank_zscore_gap(cand_off, n_cand, x, out):
    """Fill ``out[:, 3c:3c+3]`` with rank, z-score and gap-to-max of column ``c`` of ``x``.

    Rank: 0 = lowest, 1 = highest, ties share their average rank. Mean, std and
    max skip NaN.
    """
    n_col = x.shape[1]
    for f in nb.prange(len(n_cand)):
        co = cand_off[f]
        m = n_cand[f]
        for c in range(n_col):
            v = x[co:co + m, c]
            order = np.argsort(v)
            r = np.empty(m, np.float32)
            i = 0
            while i < m:
                j = i
                while j + 1 < m and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            mu = 0.0
            count = 0
            mx = -1e30
            for k in range(m):
                if not np.isnan(v[k]):
                    mu += v[k]
                    count += 1
                    if v[k] > mx:
                        mx = v[k]
            mu = mu / count if count > 0 else 0.0
            sd = 0.0
            for k in range(m):
                if not np.isnan(v[k]):
                    sd += (v[k] - mu) ** 2
            sd = np.sqrt(sd / count) if count > 0 else 0.0
            for k in range(m):
                out[co + k, 3 * c] = r[k] / (m - 1) if m > 1 else 0.5
                out[co + k, 3 * c + 1] = (v[k] - mu) / (sd + 1e-6)
                out[co + k, 3 * c + 2] = v[k] - mx


def feed_max(cand_off: np.ndarray, n_cand: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The feed maximum of ``x`` repeated for every candidate of the feed."""
    return np.repeat(np.maximum.reduceat(x, cand_off), n_cand)
