"""The probability that a train feed was shown in random order.

What it measures. Val and test feeds were shown to users in random order.
Train is a mix: most feeds were ordered by the production ranker, some were
random-order feeds like val and test, and the data does not say which. This
module estimates, for every train feed, the probability ``p_random`` that it
was a random-order feed. The code names these feeds "exploration" feeds, as
the lab did. The trainer uses the probability to tell the ranker which regime
a feed belongs to and to weight the position-imitation loss.

How. Two observations separate the regimes.

1. In a production feed the shown position follows the production ranker's
   score, so a model that predicts position from the candidate's inputs
   correlates with the true position. We fit such a proxy (LightGBM, target:
   the normalised rank of the position inside the feed) and compute, per
   feed, the Spearman correlation ``rho`` between its prediction and the true
   position. Under random order ``z = rho * sqrt(M - 1)`` is standard normal;
   under production order it is shifted to the right.
2. Production feeds have more holes in their position sequence (candidates
   that were shown but are not in the data) than random-order feeds. The
   ``gap_share`` of a feed is the share of missing positions below its last
   one.

Per feed-size bin we fit a two-component mixture over ``(z, log(gap_share +
0.01))`` by EM, with the ``z`` of the random component fixed to N(0, 1), and
take the posterior of the random component as ``p_random``.

The 400,000 feeds used to fit the proxy (200,000 for an item-level encoding of
position, 200,000 for the trees) get no ``z`` and no ``p_random`` (NaN): the
proxy has seen their positions.

Why it is safe. Only train feeds are scored, and the quantity is a property
of how a feed was served, read from positions, which exist in train only. It
is not a candidate input: val and test rows are always treated as random-order
feeds. Clicks are not used.

Reproducibility. Reading the stored proxy (``exploration.position_proxy``)
reproduces every stored value; a refit is not run-to-run reproducible, see
docs/10-reproduce.md. ``PROXY_PARAMS`` keeps the settings of the stored fit.

Run time on 12 cores: 2 minutes to fit the proxy (its 40 threads share the
cores), 10 minutes to score the remaining 2.75M feeds, under a minute for the
mixture.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import lightgbm as lgb
import numba as nb
import numpy as np
import polars as pl
from scipy.stats import norm

from avito_ranker.data.flatten import feed_offsets, read_feeds, read_flat, read_item_embeddings, rows_of_feeds
from avito_ranker.io import clear_for_rewrite

PROXY_SAMPLE_FEEDS = 600_000  # feeds drawn for the proxy; the first two thirds are used
PROXY_SAMPLE_SEED = 11
ENCODING_FEEDS = 200_000  # item-level position encoding
TREE_FEEDS = 200_000  # LightGBM fit
PROXY_ROUNDS = 400
# The thread count and the histogram layout set the order in which LightGBM adds up gradients,
# so they are part of the recipe: 40 threads and row-wise histograms are what the stored proxy was
# fitted with (docs/10-reproduce.md has the refit measurements). The 40 threads also run on fewer cores.
PROXY_PARAMS = {
    "objective": "regression", "learning_rate": 0.1, "num_leaves": 255, "min_data_in_leaf": 200,
    "feature_fraction": 0.9, "bagging_fraction": 0.5, "bagging_freq": 1,
    "num_threads": 40, "force_row_wise": True, "verbose": -1,
}
PROXY_FEATURES = (
    "f1", "f2", "f3", "f4", "f5", "category_id", "dot", "cos", "inorm",
    "hcat_cnt", "hcat_share", "hcat_recent", "hcat_cnt_ne1",
    "f1_r", "f3_r", "f4_r", "f5_r", "dot_r", "M", "H",
    "hsim_max", "hsim_mean", "hsim_max_ne1", "item_prank", "item_n",
)
SCORING_CHUNK_FEEDS = 300_000
HISTORY_ITEMS = 32  # newest history items compared with a candidate
FEED_SIZE_BINS = (5, 10, 20, 35, 50, 75, 100, 150, 200, 250, 301)
EM_ITERATIONS = 500
EM_STD_FLOOR = 0.05  # without it the fit collapses for feeds of 10 to 49 candidates


# ----------------------------------------------------------------------------- feed selection and gathering
def proxy_feed_sample() -> np.ndarray:
    """The sorted train feeds drawn for the proxy. The first ``ENCODING_FEEDS + TREE_FEEDS`` are used."""
    n_feeds = len(read_feeds("train"))
    rng = np.random.default_rng(PROXY_SAMPLE_SEED)
    return np.sort(rng.choice(n_feeds, size=PROXY_SAMPLE_FEEDS, replace=False))


# ----------------------------------------------------------------------------- inputs of the position proxy
@nb.njit(parallel=True, cache=True)
def _user_similarity(user_emb, feed_of_cand, table, emb_idx):
    """Dot product and cosine between user and item embedding, and the item norm."""
    n = len(emb_idx)
    dot = np.zeros(n, np.float32)
    cos = np.zeros(n, np.float32)
    inorm = np.zeros(n, np.float32)
    for i in nb.prange(n):
        e = emb_idx[i]
        if e < 0:
            dot[i] = np.nan
            cos[i] = np.nan
            inorm[i] = np.nan
            continue
        u = user_emb[feed_of_cand[i]]
        s = 0.0
        nu = 0.0
        ne = 0.0
        for j in range(128):
            a = float(u[j])
            b = float(table[e, j])
            s += a * b
            nu += a * a
            ne += b * b
        dot[i] = s
        inorm[i] = np.sqrt(ne)
        cos[i] = s / np.sqrt(nu * ne) if nu > 0 and ne > 0 else 0.0
    return dot, cos, inorm


@nb.njit(parallel=True, cache=True)
def _history_category_counts(hist_off, n_hist, hist_cat, hist_type, cand_off, n_cand, cand_cat):
    """Per candidate: history events in its category (all, share, among the newest 20, of type other than 1)."""
    n_feeds = len(n_cand)
    n = cand_off[-1] + n_cand[-1] if n_feeds > 0 else 0
    cnt = np.zeros(n, np.float32)
    share = np.zeros(n, np.float32)
    recent = np.zeros(n, np.float32)
    cnt_ne1 = np.zeros(n, np.float32)
    for f in nb.prange(n_feeds):
        c = np.zeros(64, np.int32)
        r = np.zeros(64, np.int32)
        c2 = np.zeros(64, np.int32)
        ho = hist_off[f]
        for j in range(n_hist[f]):
            k = hist_cat[ho + j]
            c[k] += 1
            if j < 20:
                r[k] += 1
            if hist_type[ho + j] != 1:
                c2[k] += 1
        for j in range(n_cand[f]):
            k = cand_cat[cand_off[f] + j]
            cnt[cand_off[f] + j] = c[k]
            share[cand_off[f] + j] = c[k] / max(n_hist[f], 1)
            recent[cand_off[f] + j] = r[k]
            cnt_ne1[cand_off[f] + j] = c2[k]
    return cnt, share, recent, cnt_ne1


@nb.njit(parallel=True, cache=True)
def _history_similarity(cand_off, n_cand, cand_emb, hist_off, n_hist, hist_emb, hist_type, table, n_items):
    """Per candidate: max and mean cosine to the newest history items, and the max over events of type other than 1."""
    n = cand_off[-1] + n_cand[-1]
    mx = np.full(n, np.nan, np.float32)
    mean = np.full(n, np.nan, np.float32)
    mx2 = np.full(n, np.nan, np.float32)
    for f in nb.prange(len(n_cand)):
        hv = np.zeros((n_items, 128), np.float32)
        ht = np.zeros(n_items, np.int64)
        nh = 0
        for j in range(n_hist[f]):
            e = hist_emb[hist_off[f] + j]
            if e < 0:
                continue
            nn = 0.0
            for d in range(128):
                v = float(table[e, d])
                hv[nh, d] = v
                nn += v * v
            if nn <= 0:
                continue
            nn = np.sqrt(nn)
            for d in range(128):
                hv[nh, d] /= nn
            ht[nh] = hist_type[hist_off[f] + j]
            nh += 1
            if nh >= n_items:
                break
        if nh == 0:
            continue
        for i in range(n_cand[f]):
            e = cand_emb[cand_off[f] + i]
            if e < 0:
                continue
            cv = np.empty(128, np.float32)
            nn = 0.0
            for d in range(128):
                v = float(table[e, d])
                cv[d] = v
                nn += v * v
            if nn <= 0:
                continue
            nn = np.sqrt(nn)
            best = -2.0
            s = 0.0
            best2 = -2.0
            for h in range(nh):
                c = 0.0
                for d in range(128):
                    c += hv[h, d] * cv[d]
                c /= nn
                s += c
                if c > best:
                    best = c
                if ht[h] != 1 and c > best2:
                    best2 = c
            mx[cand_off[f] + i] = best
            mean[cand_off[f] + i] = s / nh
            if best2 > -2.0:
                mx2[cand_off[f] + i] = best2
    return mx, mean, mx2


def proxy_inputs(sel: np.ndarray) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Candidate table of the given train feeds with everything the proxy reads except the item encoding.

    Returns ``(frame, cand_off, n_cand)`` with offsets local to the selection.
    The frame also carries ``item_id``, ``pos`` and ``fid`` (the feed's index
    inside the selection).
    """
    feeds = read_feeds("train")
    n_cand = feeds["M"].to_numpy()[sel]
    n_hist = feeds["H"].to_numpy()[sel]
    rows = rows_of_feeds(feeds["cand_off"].to_numpy()[sel], n_cand)
    names = ("f1", "f2", "f3", "f4", "f5", "category_id", "emb_idx", "item_id", "pos")
    cand = {name: np.asarray(read_flat("train", f"cand_{name}")[rows]) for name in names}
    cand_off = feed_offsets(n_cand)
    fid = np.repeat(np.arange(len(sel)), n_cand)

    columns = {name: cand[name] for name in names if name != "emb_idx"}
    columns["fid"] = fid.astype(np.int32)
    columns["M"] = np.repeat(n_cand, n_cand).astype(np.int32)
    columns["H"] = np.repeat(n_hist, n_cand).astype(np.int32)

    table = read_item_embeddings("train")
    user_emb = np.asarray(read_flat("train", "user_emb")[sel])
    cand_emb = cand["emb_idx"].astype(np.int64)
    columns["dot"], columns["cos"], columns["inorm"] = _user_similarity(user_emb, fid, table, cand_emb)

    hist_rows = rows_of_feeds(feeds["hist_off"].to_numpy()[sel], n_hist)
    hist_cat = np.asarray(read_flat("train", "hist_category_id")[hist_rows]).astype(np.int64)
    hist_type = np.asarray(read_flat("train", "hist_event_type")[hist_rows]).astype(np.int64)
    hist_emb = np.asarray(read_flat("train", "hist_emb_idx")[hist_rows]).astype(np.int64)
    hist_off = feed_offsets(n_hist)
    n_cand64, n_hist64 = n_cand.astype(np.int64), n_hist.astype(np.int64)
    (columns["hcat_cnt"], columns["hcat_share"], columns["hcat_recent"], columns["hcat_cnt_ne1"]) = \
        _history_category_counts(hist_off, n_hist64, hist_cat, hist_type, cand_off, n_cand64,
                                 cand["category_id"].astype(np.int64))

    frame = pl.DataFrame(columns)
    frame = frame.with_columns([
        (pl.col(k).rank("average").over("fid") / pl.col("M")).alias(f"{k}_r") for k in ("f1", "f3", "f4", "f5", "dot")
    ])
    hsim_max, hsim_mean, hsim_max_ne1 = _history_similarity(
        cand_off, n_cand64, cand_emb, hist_off, n_hist64, hist_emb, hist_type, table, HISTORY_ITEMS)
    frame = frame.with_columns(
        pl.Series("hsim_max", hsim_max), pl.Series("hsim_mean", hsim_mean), pl.Series("hsim_max_ne1", hsim_max_ne1))
    return frame, cand_off, n_cand


def _position_rank(frame: pl.DataFrame) -> pl.DataFrame:
    """Add ``prank``: the rank of the position inside the feed, scaled to [0, 1]."""
    return frame.with_columns(((pl.col("pos").rank("ordinal").over("fid") - 1) / (pl.col("M") - 1)).alias("prank"))


def item_position_encoding(sample: np.ndarray) -> pl.DataFrame:
    """Mean position rank and impression count per item over the first ``ENCODING_FEEDS`` feeds of the sample."""
    sel = sample[:ENCODING_FEEDS]
    feeds = read_feeds("train")
    n_cand = feeds["M"].to_numpy()[sel]
    rows = rows_of_feeds(feeds["cand_off"].to_numpy()[sel], n_cand)
    frame = pl.DataFrame({
        "item_id": np.asarray(read_flat("train", "cand_item_id")[rows]),
        "pos": np.asarray(read_flat("train", "cand_pos")[rows]),
        "fid": np.repeat(np.arange(len(sel)), n_cand),
        "M": np.repeat(n_cand, n_cand),
    })
    return _position_rank(frame).group_by("item_id").agg(
        pl.col("prank").mean().alias("item_prank"), pl.len().alias("item_n"))


def _matrix(frame: pl.DataFrame, encoding: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    """Join the item encoding to the candidate table and return it with the proxy's input matrix."""
    frame = frame.join(encoding, on="item_id", how="left").sort("fid", maintain_order=True)
    return frame, frame.select(list(PROXY_FEATURES)).to_numpy().astype(np.float32)


def fit_position_proxy() -> lgb.Booster:
    """Fit the model that predicts a candidate's position rank from its inputs.

    Uses the settings of ``PROXY_PARAMS`` whatever the thread budget of the
    caller; restrict the process to fewer cores from outside if needed.
    """
    sample = proxy_feed_sample()
    encoding = item_position_encoding(sample)
    frame, _, _ = proxy_inputs(sample[ENCODING_FEEDS:ENCODING_FEEDS + TREE_FEEDS])
    frame, matrix = _matrix(_position_rank(frame), encoding)
    data = lgb.Dataset(matrix, frame["prank"].to_numpy(), feature_name=list(PROXY_FEATURES),
                       categorical_feature=["category_id"])
    return lgb.train(PROXY_PARAMS, data, num_boost_round=PROXY_ROUNDS)


# ----------------------------------------------------------------------------- per-feed statistics
@nb.njit(cache=True)
def _average_rank(x):
    """Rank of each value of ``x`` from 0, tied values sharing the mean of their ranks."""
    n = len(x)
    idx = np.argsort(x, kind="mergesort")
    r = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        v = 0.5 * (i + j)
        for k in range(i, j + 1):
            r[idx[k]] = v
        i = j + 1
    return r


@nb.njit(cache=True)
def _pearson(a, b):
    """Pearson correlation of two arrays; NaN when either is constant."""
    n = len(a)
    ma = a.mean()
    mb = b.mean()
    sab = 0.0
    saa = 0.0
    sbb = 0.0
    for i in range(n):
        da = a[i] - ma
        db = b[i] - mb
        sab += da * db
        saa += da * da
        sbb += db * db
    if saa <= 0 or sbb <= 0:
        return np.nan
    return sab / np.sqrt(saa * sbb)


@nb.njit(parallel=True, cache=True)
def feed_spearman(position, predicted, cand_off, n_cand):
    """Spearman correlation between two candidate columns, one value per feed (NaN if either is constant)."""
    rho = np.full(len(n_cand), np.nan)
    for f in nb.prange(len(n_cand)):
        o = cand_off[f]
        m = n_cand[f]
        rho[f] = _pearson(_average_rank(position[o:o + m].astype(np.float64)),
                          _average_rank(predicted[o:o + m].astype(np.float64)))
    return rho


@nb.njit(parallel=True, cache=True)
def position_gaps(position, cand_off, n_cand):
    """Per feed: how many positions below the last shown one are missing from the data."""
    n_gap = np.zeros(len(n_cand))
    for f in nb.prange(len(n_cand)):
        o = cand_off[f]
        m = n_cand[f]
        last = 0
        for j in range(o, o + m):
            if position[j] > last:
                last = position[j]
        present = np.zeros(last + 1, np.bool_)
        for j in range(o, o + m):
            present[position[j]] = True
        missing = 0
        for p in range(last + 1):
            if not present[p]:
                missing += 1
        n_gap[f] = missing
    return n_gap


def order_statistic(proxy: lgb.Booster, threads: int = 12, log: Callable[[str], None] | None = None) -> np.ndarray:
    """``z = rho * sqrt(M - 1)`` per train feed; NaN for the feeds the proxy was fitted on."""
    n_feeds = len(read_feeds("train"))
    sample = proxy_feed_sample()
    encoding = item_position_encoding(sample)
    to_score = np.setdiff1d(np.arange(n_feeds), sample[:ENCODING_FEEDS + TREE_FEEDS])
    z = np.full(n_feeds, np.nan)
    for start in range(0, len(to_score), SCORING_CHUNK_FEEDS):
        sel = to_score[start:start + SCORING_CHUNK_FEEDS]
        frame, cand_off, n_cand = proxy_inputs(sel)
        frame, matrix = _matrix(frame, encoding)
        predicted = proxy.predict(matrix, num_threads=threads)
        rho = feed_spearman(frame["pos"].to_numpy().astype(np.float64), predicted.astype(np.float64),
                            cand_off.astype(np.int64), n_cand.astype(np.int64))
        z[sel] = rho * np.sqrt(n_cand - 1)
        if log is not None:
            log(f"scored {min(start + SCORING_CHUNK_FEEDS, len(to_score)):,} of {len(to_score):,} feeds")
    return z


def gap_share() -> np.ndarray:
    """Share of missing positions per train feed: ``gaps / (gaps + M)``."""
    feeds = read_feeds("train")
    n_cand = feeds["M"].to_numpy()
    n_gap = position_gaps(np.asarray(read_flat("train", "cand_pos")).astype(np.int64),
                          feeds["cand_off"].to_numpy(), n_cand)
    return n_gap / (n_gap + n_cand)


# ----------------------------------------------------------------------------- mixture
def _random_posterior(z: np.ndarray, g: np.ndarray) -> np.ndarray:
    """EM for ``pi N(z; 0, 1) N(g; ar, br) + (1 - pi) N(z; mu, s) N(g; ap, bp)``; returns P(random | z, g)."""
    pi = 0.15
    mu, s = np.median(z) + 0.5, np.std(z)
    ar, br = np.quantile(g, 0.2), np.std(g)
    ap, bp = np.median(g), np.std(g)
    for _ in range(EM_ITERATIONS):
        a = pi * norm.pdf(z) * norm.pdf(g, ar, br)
        b = (1 - pi) * norm.pdf(z, mu, s) * norm.pdf(g, ap, bp)
        r = a / (a + b + 1e-300)
        w = 1 - r
        pi = r.mean()
        mu = (w * z).sum() / w.sum()
        s = max(np.sqrt((w * (z - mu) ** 2).sum() / w.sum()), EM_STD_FLOOR)
        ar = (r * g).sum() / r.sum()
        br = max(np.sqrt((r * (g - ar) ** 2).sum() / r.sum()), EM_STD_FLOOR)
        ap = (w * g).sum() / w.sum()
        bp = max(np.sqrt((w * (g - ap) ** 2).sum() / w.sum()), EM_STD_FLOOR)
    return r


def mixture_posterior(z: np.ndarray, share: np.ndarray) -> np.ndarray:
    """``p_random`` per train feed from its ``z`` and gap share, one mixture per feed-size bin.

    Both statistics are rounded to float32 first. The pipeline that produced
    the stored probabilities kept them in float32 between two steps, and the
    posterior is reproduced bit for bit only on the rounded values.
    """
    z = z.astype(np.float32).astype(np.float64)
    log_share = np.log(share.astype(np.float32).astype(np.float64) + 0.01)
    n_cand = read_feeds("train")["M"].to_numpy()
    posterior = np.full(len(n_cand), np.nan)
    for lo, hi in pairwise(FEED_SIZE_BINS):
        members = (n_cand >= lo) & (n_cand < hi) & ~np.isnan(z)
        posterior[np.where(members)[0]] = _random_posterior(z[members], log_share[members])
    return posterior


def exploration_probability(proxy: lgb.Booster, threads: int = 12,
                            log: Callable[[str], None] | None = None) -> dict[str, np.ndarray]:
    """Everything the trainer reads: ``fidx``, ``p_random``, ``z`` and ``gap_share`` per train feed."""
    z = order_statistic(proxy, threads, log)
    share = gap_share()
    posterior = mixture_posterior(z, share)
    return {
        "fidx": np.arange(len(z)),
        "p_random": posterior.astype(np.float32),
        "z": z.astype(np.float32),
        "gap_share": share.astype(np.float32),
    }


def save_exploration_probability(target: Path, arrays: dict[str, np.ndarray], overwrite: bool = False) -> None:
    """Write the arrays as one ``.npz``; refuses to replace an existing file unless told to.

    With ``overwrite`` the old file is unlinked first, so a hard-linked copy
    elsewhere keeps its stored values.
    """
    clear_for_rewrite(target, overwrite)
    with open(target, "xb") as handle:
        np.savez(handle, **arrays)
