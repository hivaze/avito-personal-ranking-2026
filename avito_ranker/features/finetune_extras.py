"""The 42 extra inputs of the fine-tune stage.

After training on train, every ensemble member is fine-tuned on the
validation days with 5 folds grouped by user. That stage gets 42 extra
candidate inputs through a small zero-initialised MLP. They exist for val and
test only, so they can use constructions that cannot be made leak-free on
train (encoder scores from the ``all`` and ``no_test`` models) or that the
train stage reads in another form.

By family:

``centrality``                    cosine to the mean candidate of the feed
``it_hist_*`` (14)                leave-user-out occurrence counts: all events with its within-feed rank,
                                  signed-log z-score and gap to the feed maximum; event types 1 to 10
``seq_cv_*`` (3)                  co-visitation: weighted score, its within-feed rank, scanned occurrences
``seq_max_all``, ``seq_top3``, ``seq_argmax_hpos``
                                  best content match in the whole history, mean of the three best,
                                  and how far back the best match is
``ctx_nd_*`` (10)                 neighbours among the candidates (same arrays as in the train stage)
``pre_*`` (11)                    trajectory-encoder scores made with the partial content table: both
                                  head scores and their z, the score without the ID embedding, the ID
                                  contribution, similarity to recent history in item space (32 and 8
                                  items) and in ID space, and two presence flags

Why it is safe. Every family obeys the rules of its own module: no later
snapshot of the same user, no labels of the feed itself. Statistics fitted on
train labels (item and user click rates) are not in the list, and neither is
the hour of day.

Run time on 12 threads: 2 minutes for val and test together, most of it the
co-visitation index, given the stored raw encoder scores.
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
from avito_ranker.ensemble.zscore import feed_rank_z
from avito_ranker.features import covisitation, feed_context, item_counts
from avito_ranker.features.names import FINETUNE_EXTRAS
from avito_ranker.features.store import n_candidates
from avito_ranker.features.trajectory_encoder import inputs as encoder_inputs
from avito_ranker.features.trajectory_encoder.inputs import SENTINEL
from avito_ranker.features.within_feed import feed_max, rank_zscore_gap

HISTORY_MATCH_NAMES = ("seq_max_all", "seq_top3", "seq_argmax_hpos")


@nb.njit(parallel=True, fastmath=True, cache=True)
def _history_matches(cand_off, n_cand, hist_off, n_hist, cand_emb, hist_emb, table, out):
    """Per candidate: 0 largest cosine to a history item, 1 mean of the three largest, 2 history position of
    the best match. NaN when the candidate has no embedding or the history has none."""
    dim = table.shape[1]
    for f in nb.prange(len(n_cand)):
        co = cand_off[f]
        ho = hist_off[f]
        h = n_hist[f]
        V = np.zeros((max(h, 1), dim), np.float32)
        hp = np.zeros(max(h, 1), np.int64)
        nv = 0
        for j in range(h):
            ei = hist_emb[ho + j]
            if ei < 0:
                continue
            s = 0.0
            for d in range(dim):
                x = np.float32(table[ei, d])
                V[nv, d] = x
                s += x * x
            s = np.sqrt(s) + 1e-6
            for d in range(dim):
                V[nv, d] /= s
            hp[nv] = j
            nv += 1
        for i in range(n_cand[f]):
            r = co + i
            ei = cand_emb[r]
            if ei < 0 or nv == 0:
                out[r, 0] = np.nan
                out[r, 1] = np.nan
                out[r, 2] = np.nan
                continue
            c = np.zeros(dim, np.float32)
            s = 0.0
            for d in range(dim):
                x = np.float32(table[ei, d])
                c[d] = x
                s += x * x
            s = np.sqrt(s) + 1e-6
            for d in range(dim):
                c[d] /= s
            mx = -2.0
            amx = 0
            t1 = -2.0
            t2 = -2.0
            t3 = -2.0
            for k in range(nv):
                dt = 0.0
                for d in range(dim):
                    dt += c[d] * V[k, d]
                if dt > mx:
                    mx = dt
                    amx = hp[k]
                if dt > t1:
                    t3 = t2
                    t2 = t1
                    t1 = dt
                elif dt > t2:
                    t3 = t2
                    t2 = dt
                elif dt > t3:
                    t3 = dt
            out[r, 0] = mx
            if nv >= 3:
                out[r, 1] = (t1 + t2 + t3) / 3.0
            elif nv == 2:
                out[r, 1] = (t1 + t2) / 2.0
            else:
                out[r, 1] = t1
            out[r, 2] = amx


def history_matches(split: str) -> dict[str, np.ndarray]:
    """``seq_max_all``, ``seq_top3`` and ``seq_argmax_hpos`` of a split."""
    cand_off, n_cand, hist_off, n_hist = feed_layout(read_feeds(split))
    out = np.empty((n_candidates(split), 3), np.float32)
    _history_matches(cand_off, n_cand, hist_off, n_hist,
                     np.asarray(read_flat(split, "cand_emb_idx")), np.asarray(read_flat(split, "hist_emb_idx")),
                     read_item_embeddings(split), out)
    return {name: out[:, i] for i, name in enumerate(HISTORY_MATCH_NAMES)}


def count_columns(split: str, tables: item_counts.CountTables) -> dict[str, np.ndarray]:
    """The 14 occurrence-count extras of a split."""
    counts = item_counts.leave_user_out_counts(split, tables, range(item_counts.N_EVENT_TYPES + 1))
    feeds = read_feeds(split)
    cand_off = feeds["cand_off"].to_numpy()
    n_cand = feeds["M"].to_numpy().astype(np.int64)
    total = counts["it_hist_all_luo"]
    rank_z = feed_rank_z(cand_off, n_cand, total)
    z = rank_z[:, 1]
    out = {
        "it_hist_all_luo": total,
        "it_hist_all_luo_rk": rank_z[:, 0],
        "it_hist_all_luo_zsl": (np.sign(z) * np.log1p(np.abs(z))).astype(np.float32),
        "it_hist_all_luo_gap": total - feed_max(cand_off, n_cand, total),
    }
    out.update({name: values for name, values in counts.items() if name != "it_hist_all_luo"})
    return out


def covisitation_columns(split: str, covis: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The three co-visitation extras from the co-visitation inputs of the split."""
    cand_off, n_cand, _, _ = feed_layout(read_feeds(split))
    score = np.ascontiguousarray(covis["seq_cv_invd20"], dtype=np.float32)
    transformed = np.empty((len(score), 3), np.float32)
    rank_zscore_gap(cand_off, n_cand, score[:, None], transformed)
    return {"seq_cv_invd20": score, "seq_cv_invd20_rk": transformed[:, 0], "seq_cv_pop": covis["seq_cv_pop"]}


def encoder_columns(split: str) -> dict[str, np.ndarray]:
    """The 11 encoder extras of a split, from the raw scores made with the partial content table."""
    columns = encoder_inputs.eval_columns(split, "partial")
    cleaned, missing = {}, {}
    for name in ("hmax32", "hmax8"):
        values = columns[name]
        missing[name] = values == SENTINEL  # the feed has no history item to compare with
        if not (np.isfinite(values).all() and values[~missing[name]].min() > -1.5):
            raise ValueError(f"{split}: unexpected values in the encoder column {name}")
        cleaned[name] = np.where(missing[name], np.nan, values).astype(np.float32)
    return {
        "pre_fe": columns["fe"],
        "pre_fe_z": columns["fe_z"],
        "pre_new": columns["new"],
        "pre_new_z": columns["new_z"],
        "pre_hmax32_c": cleaned["hmax32"],
        "pre_hmax8_c": cleaned["hmax8"],
        "pre_fe_noid": columns["fe_noid"],
        "pre_id_contrib_v2": encoder_inputs.id_contribution(split),
        "pre_idmax32": columns["idmax32"],
        "pre_idmax32_has": np.isfinite(columns["idmax32"]).astype(np.float32),
        "pre_hmax32_sent": missing["hmax32"].astype(np.float32),
    }


def build_finetune_extras(splits: tuple[str, ...] = ("val", "test")) -> dict[str, dict[str, np.ndarray]]:
    """All 42 extras for val and/or test, keyed by split then by file name (see ``FINETUNE_EXTRAS``)."""
    tables = item_counts.load_count_tables()
    covis = covisitation.build_eval(splits)
    result = {}
    for split in splits:
        base = feed_context.base_similarity(split)
        columns = {"centrality": base["centrality"]}
        columns.update(count_columns(split, tables))
        columns.update(covisitation_columns(split, covis[split]))
        columns.update(history_matches(split))
        columns.update(feed_context.build_neighbours(split, base))
        columns.update(encoder_columns(split))
        result[split] = {name: columns[name] for name in FINETUNE_EXTRAS}
    return result
