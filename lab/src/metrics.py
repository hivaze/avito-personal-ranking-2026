"""NDCG exactly as the organizers' evaluator: binary gain, log2 discount, pessimistic ties,
feeds without positives -> 1, unweighted mean over feeds."""
import numpy as np


def ndcg_per_feed(scores, targets, offsets, M, k=None):
    """scores/targets flat arrays in feed order; offsets/M per feed. Returns per-feed NDCG."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n = len(M)
    fid = np.repeat(np.arange(n), M)
    # sort by feed, then score desc, then target asc (pessimistic ties: negatives first)
    order = np.lexsort((targets, -scores, fid))
    t_sorted = targets[order]
    rank = np.arange(len(order)) - np.repeat(offsets, M)  # 0-based position within feed
    disc = 1.0 / np.log2(rank + 2.0)
    if k is not None:
        disc = np.where(rank < k, disc, 0.0)
    dcg = np.bincount(fid, weights=t_sorted * disc, minlength=n)
    npos = np.bincount(fid, weights=targets, minlength=n)
    ideal_t = (rank < np.repeat(npos, M)).astype(np.float64)
    idcg = np.bincount(fid, weights=ideal_t * disc, minlength=n)
    out = np.ones(n)
    m = idcg > 0
    out[m] = dcg[m] / idcg[m]
    return out


def ndcg(scores, targets, offsets, M, k=None):
    return float(ndcg_per_feed(scores, targets, offsets, M, k).mean())
