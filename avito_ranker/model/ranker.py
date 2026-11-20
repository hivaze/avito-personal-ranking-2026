"""The ranker: scores every candidate of a feed against the user's history and against the other candidates.

Tokens, all of width ``d``:

* user token: a linear map of the unit user embedding;
* history tokens: a linear map of the item's unit content embedding plus
  embeddings of event type, event source, category and recency index
  (0 = newest), encoded by one self-attention block over [user, history];
* candidate tokens: a linear map of [content embedding, content embedding x
  user embedding], plus the category embedding and an MLP of the dense block
  (``avito_ranker.model.dense``), plus a domain embedding, plus the retrieval
  summary of the candidate's nearest history items.

Each of the two main blocks lets the candidates attend to [user, history]
(with the content cosine as a per-head attention prior) and then to each other.
The candidates carry no position: val and test feeds are shown in random
order, so the model must not learn one.

The score is an MLP of the final token plus a linear function of the dense
block. A second head of the same shape learns the order in which the
production system showed the candidates; it is an auxiliary target in training
and is never used at inference.

The domain embedding has three rows. Train feeds are a mix of row 0
(production order) and row 1 (feeds that look randomly ordered), weighted by
the probability that the feed is an exploration feed. Val and test feeds are
all randomly ordered: a train-only model scores them with row 1, and the
fine-tune gives them row 2, which starts as a copy of row 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from avito_ranker.model.batch import Batch
from avito_ranker.model.blocks import HistoryRetrieval, TransformerBlock, full_precision, side_stream
from avito_ranker.model.config import (
    EMBEDDING_DIM,
    MAX_HISTORY,
    N_CATEGORIES,
    N_EVENT_SOURCES,
    N_EVENT_TYPES,
    RankerConfig,
)
from avito_ranker.model.dense import dense_features

PRODUCTION_ORDER, RANDOM_ORDER, VALIDATION_DAYS = 0, 1, 2  # rows of the domain embedding


def _score_head(d: int) -> nn.Sequential:
    """An MLP from a token to one score."""
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))


class Ranker(nn.Module):
    """Feed ranker built from a :class:`~avito_ranker.model.config.RankerConfig`.

    The order in which the layers are created below is the order in which they
    draw their initial weights. It is kept as in the trained members, so that a
    seed gives the same starting point.
    """

    def __init__(self, config: RankerConfig):
        super().__init__()
        self.config = config
        d, n_dense = config.d_model, config.n_dense
        block = dict(d=d, n_heads=config.n_heads, dropout=config.dropout, ffn_multiplier=config.ffn_multiplier)

        # Standardisation of the dense block, measured once on train batches.
        self.register_buffer("dense_mean", torch.zeros(n_dense))
        self.register_buffer("dense_std", torch.ones(n_dense))

        self.category = nn.Embedding(N_CATEGORIES, d)  # shared by candidates and history events
        self.candidate_content = nn.Linear(2 * EMBEDDING_DIM, d)
        self.candidate_missing_vector = nn.Parameter(torch.zeros(d))  # added when the item has no content vector
        self.dense_mlp = nn.Sequential(nn.Linear(n_dense, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.candidate_norm = nn.LayerNorm(d)
        with side_stream(1):
            self.retrieval = HistoryRetrieval(d, config.retrieval_slots, config.dropout)
        self.inference_domain = RANDOM_ORDER
        self.domain = nn.Embedding(3, d)
        nn.init.normal_(self.domain.weight, std=0.02)

        self.user_content = nn.Linear(EMBEDDING_DIM, d)
        self.history_content = nn.Linear(EMBEDDING_DIM, d)
        self.history_missing_vector = nn.Parameter(torch.zeros(d))
        self.event_type = nn.Embedding(N_EVENT_TYPES + 1, d)
        self.event_source = nn.Embedding(N_EVENT_SOURCES, d)
        self.recency = nn.Embedding(MAX_HISTORY, d)
        self.history_norm = nn.LayerNorm(d)

        # Side streams 2, 4, .. belonged to a construction step whose draws were
        # discarded, so the gates of the attention blocks use 3, 5, 7, ..
        gate_streams = iter(range(3, 3 + 2 * (config.n_history_layers + 2 * config.n_blocks), 2))
        self.history_encoder = nn.ModuleList(
            TransformerBlock(**block, gate_stream=next(gate_streams)) for _ in range(config.n_history_layers))
        self.cross_blocks = nn.ModuleList(
            TransformerBlock(**block, gate_stream=next(gate_streams), similarity_bias=True)
            for _ in range(config.n_blocks))
        self.set_blocks = nn.ModuleList(
            TransformerBlock(**block, gate_stream=next(gate_streams)) for _ in range(config.n_blocks))

        self.head = _score_head(d)
        self.position_head = _score_head(d)
        self.wide = nn.Linear(n_dense, 1)

        self.finetune_mlp = None
        if config.n_finetune_extras:
            # The last layer starts at zero: when the fine-tune begins, the
            # model with the branch scores exactly like the model without it.
            self.finetune_mlp = nn.Sequential(nn.Linear(config.n_finetune_extras, 2 * d), nn.GELU(),
                                              nn.Linear(2 * d, d))
            nn.init.zeros_(self.finetune_mlp[2].weight)
            nn.init.zeros_(self.finetune_mlp[2].bias)

    def set_dense_scaling(self, mean: Tensor, std: Tensor) -> None:
        """Store the mean and standard deviation used to standardise the dense block."""
        self.dense_mean.copy_(mean)
        self.dense_std.copy_(std.clamp_min(1e-3))

    def use_validation_domain(self) -> None:
        """Prepare the domain embedding for the fine-tune on the validation days.

        Row 2 becomes a copy of the random-order row and is used from now on
        for every batch that carries no domain mix, that is for val and test.
        """
        with torch.no_grad():
            self.domain.weight[VALIDATION_DAYS] = self.domain.weight[RANDOM_ORDER]
        self.inference_domain = VALIDATION_DAYS

    def dense_features(self, batch: Batch) -> Tensor:
        """The raw dense block of a batch for this model's view."""
        return dense_features(batch, self.config.view)

    def _domain_vector(self, batch: Batch) -> Tensor:
        """Each feed's mix of the domain rows ``[B, 1, d]``; without a mix, the inference domain."""
        mix = batch.domain_mix
        if mix is None:
            n_feeds = batch.candidate_mask.shape[0]
            mix = F.one_hot(torch.full((n_feeds,), self.inference_domain, device=batch.device), 3).float()
        return (mix.float() @ self.domain.weight)[:, None, :]

    def _context(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Encoded [user, history] tokens ``[B, 1 + K, d]`` and their mask."""
        history = batch.history_unit
        n_feeds, n_events = history.shape[:2]
        user = self.user_content(batch.user_unit)[:, None, :]
        tokens = self.history_content(history) + self.event_type(batch.history_event_type) \
            + self.event_source(batch.history_event_source) + self.category(batch.history_category) \
            + self.recency(torch.arange(n_events, device=history.device))[None] \
            + (~batch.history_has_embedding)[..., None].float() * self.history_missing_vector
        tokens = self.history_norm(tokens)
        context = torch.cat([user, tokens], 1)
        mask = torch.cat([torch.ones(n_feeds, 1, dtype=torch.bool, device=history.device), batch.history_mask], 1)
        for block in self.history_encoder:
            context = block(context, mask)
        return context, mask

    def forward(self, batch: Batch, dense: Tensor | None = None,
                with_position_scores: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Score the candidates of a batch.

        ``dense`` is the raw dense block. The trainer computes it in float32
        before it enters mixed precision and passes it in; when it is omitted
        the block is computed here, under whatever precision is active.

        Returns float32 scores ``[B, M]`` (higher is better; padded slots hold
        meaningless values). With ``with_position_scores`` it returns the pair
        (scores, scores of the production-position head).
        """
        if dense is None:
            dense = self.dense_features(batch)
        dense = ((dense - self.dense_mean) / self.dense_std).clamp(-6, 6)
        candidate, user = batch.candidate_unit, batch.user_unit

        token = self.candidate_content(torch.cat([candidate, candidate * user[:, None, :]], -1))
        token = token + self.category(batch.candidate_category) + self.dense_mlp(dense) \
            + batch.candidate_no_embedding[..., None].float() * self.candidate_missing_vector
        token = self.candidate_norm(token)
        domain = self._domain_vector(batch)
        token = token + domain
        if self.finetune_mlp is not None:
            if batch.finetune_extras is None:
                raise ValueError("this model has a fine-tune branch; attach the extras to the split "
                                 "(SplitTensors.attach_finetune_extras)")
            token = token + self.finetune_mlp(batch.finetune_extras)

        context, context_mask = self._context(batch)

        cosine = torch.einsum("bmd,bkd->bmk", candidate, batch.history_unit) * batch.history_has_embedding[:, None, :]
        token = token + self.retrieval(candidate, batch.history_unit, batch.history_has_embedding,
                                       batch.history_event_type, batch.history_event_source, cosine,
                                       batch.candidate_no_embedding)
        # Attention prior of the cross blocks: cosine to the user (key 0) and to each history item.
        prior = torch.cat([torch.einsum("bmd,bd->bm", candidate, user)[..., None], cosine], -1)
        for cross, among_candidates in zip(self.cross_blocks, self.set_blocks):
            token = cross(token, context_mask, context, prior)
            token = among_candidates(token, batch.candidate_mask)
        token = token + domain

        # The heads run in float32. In bfloat16 the scores are quantised, equal
        # scores appear inside a feed, and the metric orders ties pessimistically.
        # Moving the heads to float32 took an early model from 0.3705 to 0.3736
        # validation NDCG.
        with full_precision(token):
            score = self.head(token.float()).squeeze(-1) + self.wide(dense.float()).squeeze(-1)
        if with_position_scores:
            return score.float(), self.position_head(token).squeeze(-1).float()
        return score.float()
