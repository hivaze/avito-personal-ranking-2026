"""The losses of a member's training and fine-tune, and the distillation target.

Both losses are listwise: the scores of a feed's candidates go through a
softmax over the feed, and the loss is the cross-entropy between that
distribution and a target distribution over the same candidates. This matches
the task, which only asks for an order inside each feed.

* The main loss pulls the score distribution toward the clicks. For a
  distilled student the target mixes the clicks with the teacher's softmax
  (:func:`distillation_target`).
* The auxiliary loss trains a second head to reproduce the order in which the
  production system showed the feed. It is weighted per feed by ``1 - p``,
  where ``p`` is the probability that the feed was an exploration feed: a
  feed shown in random order carries no production order to learn.
* The fine-tune on the validation days uses the main loss on the clicks with
  another normalisation, the plain mean over the feeds of the batch that have
  a click (:func:`mean_listwise_loss`).

Tensors are padded ``[B, M]`` batches; ``mask`` marks the real candidates.
Padded slots get a logit of -1e4 so that they take no probability mass.

The order of the floating-point operations below is that of the code that
trained the competition members; a seeded run reproduces their loss only if
it stays so.
"""
from __future__ import annotations

import torch
from torch import Tensor

PADDING_LOGIT = -1e4


def distillation_target(clicks: Tensor, teacher_scores: Tensor, mask: Tensor, teacher_weight: float) -> Tensor:
    """Target distribution ``(1 - w) * clicks / n_clicks + w * softmax(teacher)`` for each feed.

    ``teacher_scores`` are the teacher's logits for the batch's candidates.
    Each feed with a click gets a target that sums to 1. A feed without a
    click gets an all-zero target, so the main loss ignores it, as it ignores
    such feeds when training on clicks alone.
    """
    teacher = torch.softmax(teacher_scores.masked_fill(~mask, PADDING_LOGIT), 1) * mask
    n_clicks = clicks.sum(1, keepdim=True)
    return (1 - teacher_weight) * clicks / n_clicks.clamp_min(1) + teacher_weight * teacher * (n_clicks > 0)


def listwise_loss(scores: Tensor, target: Tensor, mask: Tensor, max_feeds: int) -> Tensor:
    """Softmax cross-entropy of each feed toward ``target / target.sum()``, summed and divided by ``max_feeds``.

    ``target`` is the click label (training on clicks) or a distillation
    target. Dividing by the constant ``max_feeds`` instead of the number of
    feeds in the batch gives every feed the same weight: batches of long
    feeds hold fewer feeds, and a per-batch mean would weigh each of them
    more. Feeds whose target is all zero (no click) contribute nothing.
    """
    log_p = torch.log_softmax(scores.masked_fill(~mask, PADDING_LOGIT), dim=1)
    mass = target.sum(1)
    has_target = (mass > 0).float()
    per_feed = -(target * log_p).sum(1) / mass.clamp_min(1)
    return (per_feed * has_target).sum() / max_feeds


def mean_listwise_loss(scores: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Softmax cross-entropy of each feed toward its clicks, averaged over the feeds of the batch that have one.

    The loss of the fine-tune. The target of a feed is ``targets /
    targets.sum()``: with one click, the loss is minus the log of the softmax
    probability of the clicked candidate. Feeds without a click carry no
    signal and get weight 0.
    """
    log_probability = torch.log_softmax(scores.masked_fill(~mask, PADDING_LOGIT), dim=1)
    clicks = targets.sum(1)
    per_feed = -(targets * log_probability).sum(1) / clicks.clamp_min(1)
    weight = (clicks > 0).float()
    return (per_feed * weight).sum() / weight.sum().clamp_min(1e-6)


def production_order_loss(position_scores: Tensor, positions: Tensor, mask: Tensor, feed_weight: Tensor,
                          temperature: float, weight: float) -> Tensor:
    """``weight`` times the feed-weighted mean of the listwise cross-entropy toward the production order.

    The target of a feed is ``softmax(-position / temperature)``: the
    candidate the production system showed first gets the most mass, with a
    top-heavy decay like that of the metric. ``feed_weight`` [B] is ``1 - p``
    with ``p`` the feed's exploration probability.
    """
    target = torch.softmax((-positions.float() / temperature).masked_fill(~mask, PADDING_LOGIT), 1) * mask
    log_p = torch.log_softmax(position_scores.masked_fill(~mask, PADDING_LOGIT), 1)
    per_feed = -(target * log_p).sum(1)
    return weight * (per_feed * feed_weight).sum() / feed_weight.sum().clamp_min(1e-6)
