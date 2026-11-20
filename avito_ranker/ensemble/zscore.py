"""Within-feed standardisation of a score column.

Members of the ensemble are trained with different seeds, teachers and input
views, and the scale of their logits differs from feed to feed. Only the order
inside a feed is scored, so before any averaging each member's scores are
standardised inside each feed: ``z = (x - mean) / (std + 1e-9)``.

The kernel is the one every stored pool and submission was built with. It is
kept as it was (numba, float32 input and output) because the submission files
are compared by sha256: a float64 re-implementation in numpy changes the order
of a few near-tied candidates.
"""
from __future__ import annotations

import numba as nb
import numpy as np

from avito_ranker.ensemble.feeds import FeedIndex


@nb.njit(parallel=True, cache=True)
def feed_rank_z(offsets, sizes, x):
    """Return ``[rank, z]`` for every row, a float32 array of shape ``[len(x), 2]``.

    Column 0 is the ordinal rank inside the feed scaled to [0, 1] (0.5 in a
    feed of one candidate). Column 1 is the z-score with ``std + 1e-9`` in the
    denominator, so a constant feed gives zeros. Mean and standard deviation
    are taken in the dtype of ``x``; the stored pools and submissions were made
    with float32 input. Feeds are independent and are processed in parallel,
    and the result does not depend on the number of threads.
    """
    out = np.empty((len(x), 2), np.float32)
    for f in nb.prange(len(sizes)):
        o, m = offsets[f], sizes[f]
        v = x[o:o + m]
        r = np.argsort(np.argsort(v))
        mu = v.mean()
        sd = v.std() + 1e-9
        for k in range(m):
            out[o + k, 0] = r[k] / (m - 1) if m > 1 else 0.5
            out[o + k, 1] = (v[k] - mu) / sd
    return out


def feed_zscore(feeds: FeedIndex, x: np.ndarray) -> np.ndarray:
    """Standardise ``x`` inside each feed and return float32 z-scores.

    ``x`` is cast to float32 first, as the stored builders did.
    """
    return feed_rank_z(feeds.offsets, feeds.sizes, np.asarray(x, np.float32))[:, 1]
