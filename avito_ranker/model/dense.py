"""The dense block: one row of hand-built numbers per candidate.

The ranker reads each candidate through two channels. One is the content
embedding, which goes through attention. The other is this block: 63 columns
computed here from the raw feed, followed by the pre-computed inputs of the
member's view. The block feeds an MLP inside the candidate token and a linear
term added to the final score.

The 63 computed columns, in order:

====  =====================================================================
  15  f1..f5: the raw value, its within-feed z-score and percentile rank
   2  no-embedding flag; log(1 + norm of the content embedding)
   2  cosine and scaled dot product with the user embedding
   6  cosine to the history items: max over all, over the newest 8, over
      the newest 32; mean; a soft maximum; mean of the top 3
  10  max cosine to the history items of each event type
   2  history events in the candidate's category: log count, share
  10  the same log count for each event type
   1  recency of the newest history event in the candidate's category
   3  share of the feed in the candidate's category; log feed size;
      log(1 + history length)
  10  within-feed z-score and percentile rank of five of the columns above
   2  anonymous-session flag, alone and times log(1 + history length)
====  =====================================================================

Everything is computed on the fly from the batch, so no feature file exists
for these columns and train, val and test rows are built by the same code.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from avito_ranker.model.batch import Batch
from avito_ranker.model.config import N_COMPUTED_COLUMNS, N_EVENT_TYPES
from avito_ranker.model.inputs import InputView

if TYPE_CHECKING:  # the loader imports this package, so the import is for type checkers only
    from avito_ranker.data.loader import SplitTensors


def _zscore(x: Tensor, weight: Tensor, count: Tensor) -> Tensor:
    """Z-score of ``x`` [B, M] among the real candidates of each feed."""
    mean = (x * weight).sum(1, keepdim=True) / count
    variance = (((x - mean) * weight) ** 2).sum(1, keepdim=True) / count
    return (x - mean) / (variance.sqrt() + 1e-4)


def _percentile_rank(x: Tensor, mask: Tensor, count: Tensor) -> Tensor:
    """Rank of ``x`` [B, M] within its feed, scaled to [0, 1]; tied values share the middle rank."""
    mine = x[:, :, None]
    other = x[:, None, :]
    real = mask[:, None, :]
    below = ((other < mine) & real).sum(-1).float()
    equal = ((other == mine) & real).sum(-1).float() - 1.0
    return (below + 0.5 * equal) / (count - 1).clamp_min(1)


def _history_similarity(batch: Batch) -> tuple[list[Tensor], dict[str, Tensor]]:
    """Columns that compare the candidate's embedding with the embeddings of the history items.

    Returns the 16 columns and the three that later get within-feed versions.
    """
    has_embedding = batch.history_has_embedding  # [B, K]
    cosine = torch.einsum("bmd,bkd->bmk", batch.candidate_unit, batch.history_unit)
    absent = torch.tensor(-1.5, device=cosine.device)  # below any cosine: marks "no such event"
    masked = torch.where(has_embedding[:, None, :], cosine, absent)
    n_events = cosine.shape[-1]
    n_valid = has_embedding.float().sum(-1, keepdim=True)

    best = masked.max(-1).values
    best_newest_8 = masked[:, :, :8].max(-1).values
    best_newest_32 = masked[:, :, :32].max(-1).values
    mean = (cosine * has_embedding[:, None, :]).sum(-1) / n_valid.clamp_min(1)
    # Soft maximum with temperature 1/5, corrected for the number of events.
    soft_max = torch.logsumexp(
        torch.where(has_embedding[:, None, :], 5 * cosine, torch.tensor(-1e4, device=cosine.device)), -1) / 5
    soft_max = torch.where(n_valid > 0, soft_max - torch.log(n_valid.clamp_min(1)) / 5, absent)
    top3 = masked.topk(min(3, n_events), dim=-1).values.mean(-1)

    columns = [best, best_newest_8, best_newest_32, mean, soft_max, top3]
    event_type = batch.history_event_type
    for kind in range(1, N_EVENT_TYPES + 1):
        of_kind = has_embedding & (event_type == kind)
        columns.append(torch.where(of_kind[:, None, :], cosine, absent).max(-1).values)
    return columns, {"best": best, "soft_max": soft_max, "top3": top3}


def _category_affinity(batch: Batch) -> tuple[list[Tensor], Tensor]:
    """Columns that count the history events in the candidate's category.

    Returns the 13 columns and the log count, which later gets within-feed versions.
    """
    same_category = (batch.candidate_category[:, :, None] == batch.history_category[:, None, :]) \
        & batch.history_mask[:, None, :]
    count = same_category.float().sum(-1)
    history_length = batch.history_length.float()[:, None]
    log_count = torch.log1p(count)
    columns = [log_count, count / history_length.clamp_min(1)]
    event_type = batch.history_event_type
    for kind in range(1, N_EVENT_TYPES + 1):
        columns.append(torch.log1p((same_category & (event_type == kind)[:, None, :]).float().sum(-1)))
    # Position of the newest same-category event (0 = the user's last event), as 1 / (1 + position).
    n_events = same_category.shape[-1]
    slots = torch.arange(n_events, device=count.device)
    newest = torch.where(same_category, slots, n_events).min(-1).values.float()
    columns.append(torch.where(newest < n_events, 1.0 / (1.0 + newest), torch.zeros_like(newest)))
    return columns, log_count


def _view_columns(inputs: Tensor, view: InputView, mask: Tensor, weight: Tensor, count: Tensor) -> Tensor:
    """The pre-computed inputs as dense columns: the raw ones, then rank and z-score of the ranked ones."""
    relative = []
    for j in view.ranked:
        relative += [_percentile_rank(inputs[..., j], mask, count), _zscore(inputs[..., j], weight, count)]
    columns = inputs[..., view.raw]
    if relative:
        columns = torch.cat([columns, torch.stack(relative, -1)], -1)
    return columns


@torch.no_grad()
def dense_features(batch: Batch, view: InputView) -> Tensor:
    """Return the dense block of a batch, float32 ``[B, M, 63 + view.n_dense_columns]``.

    The values are raw: the ranker standardises them with statistics stored in
    its checkpoint. Non-finite values become 0.
    """
    mask = batch.candidate_mask
    weight = mask.float()
    count = weight.sum(1, keepdim=True)
    columns: list[Tensor] = []

    numeric = batch.candidate_numeric
    for k in range(5):
        x = numeric[..., k]
        columns += [x, _zscore(x, weight, count), _percentile_rank(x, mask, count)]

    no_embedding = batch.candidate_no_embedding.float()
    norm = batch.candidate_embedding.norm(dim=-1)
    columns += [no_embedding, torch.log1p(norm) * (1 - no_embedding)]

    cos_user = torch.einsum("bmd,bd->bm", batch.candidate_unit, batch.user_unit)
    dot_user = torch.einsum("bmd,bd->bm", batch.candidate_embedding, batch.user_embedding) / 1e4
    columns += [cos_user, dot_user]

    similarity_columns, similarity = _history_similarity(batch)
    columns += similarity_columns

    category_columns, log_category_count = _category_affinity(batch)
    columns += category_columns

    category = batch.candidate_category
    feed_share = ((category[:, :, None] == category[:, None, :]) & mask[:, None, :]).float().sum(-1) / count
    log_history = torch.log1p(batch.history_length.float()[:, None]).expand_as(feed_share)
    columns += [feed_share, torch.log(count).expand_as(feed_share), log_history]

    # A candidate is judged against the other candidates of its feed, so the
    # strongest columns also enter relative to the feed.
    for x in (cos_user, similarity["best"], similarity["soft_max"], similarity["top3"], log_category_count):
        columns += [_zscore(x, weight, count), _percentile_rank(x, mask, count)]

    # Anonymous sessions share one user id and behave differently; the two
    # columns let the model treat them as their own regime.
    anonymous = batch.is_anonymous.float()[:, None].expand_as(feed_share)
    columns += [anonymous, anonymous * log_history]

    block = torch.stack(columns, -1)
    assert block.shape[-1] == N_COMPUTED_COLUMNS
    block = torch.cat([block, _view_columns(batch.inputs, view, mask, weight, count)], -1)
    return torch.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def dense_statistics(split: SplitTensors, feed_ids: np.ndarray, view: InputView, n_batches: int = 40,
                     seed: int = 0) -> tuple[Tensor, Tensor]:
    """Mean and standard deviation of every dense column over the real candidates of a sample of batches.

    A new model is given these numbers once, before training, and keeps them in
    its checkpoint (``Ranker.set_dense_scaling``). The sample is the first
    ``n_batches`` batches of a shuffled plan of ``feed_ids``, about 20,000
    feeds.
    """
    plan = split.plan_batches(feed_ids, shuffle=True, rng=np.random.default_rng(seed))[:n_batches]
    total = total_squares = n_rows = 0
    for feeds in plan:
        batch = split.gather(feeds)
        rows = dense_features(batch, view)[batch.candidate_mask].double()
        total = total + rows.sum(0)
        total_squares = total_squares + (rows ** 2).sum(0)
        n_rows += rows.shape[0]
    mean = total / n_rows
    std = (total_squares / n_rows - mean ** 2).clamp_min(0).sqrt()
    return mean.float(), std.float()
