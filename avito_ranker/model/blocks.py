"""Building blocks of the ranker: gated attention, the transformer block, history retrieval."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from avito_ranker.model.config import EMBEDDING_DIM, N_EVENT_SOURCES, N_EVENT_TYPES


@contextlib.contextmanager
def side_stream(index: int) -> Iterator[None]:
    """Initialise a block from its own random stream and leave the main stream untouched.

    The retrieval block and the attention gates joined the model after its
    first version. So that a given seed still gave every older parameter the
    same initial value, with or without the new block, each of them draws its
    weights from a stream seeded with ``seed + 7919 + 1009 * index``. The final
    members were trained this way, and a seeded run reproduces their initial
    weights only if the streams stay as they were.

    ``torch.manual_seed`` also reseeds the GPU generators, and the fork restores
    only the CPU one. After a ranker is built, the GPU stream therefore starts
    from the seed of the last side stream; the dropout masks of a seeded
    training run depend on that.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch.initial_seed() + 7919 + 1009 * index)
        yield


def full_precision(tensor: Tensor) -> torch.autocast:
    """Context that switches mixed precision off for the device of ``tensor``."""
    return torch.autocast(tensor.device.type, enabled=False)


@contextlib.contextmanager
def tf32_matmul(enabled: bool, cudnn: bool | None = None) -> Iterator[None]:
    """Set whether float32 matrix products on the GPU may use TF32, and restore the setting afterwards.

    ``cudnn``, when given, sets cuDNN's TF32 flag the same way (the trajectory
    encoder's runs set both).
    """
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    if cudnn is not None:
        torch.backends.cudnn.allow_tf32 = cudnn
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _split_linear(fused: nn.Linear, start: int, stop: int) -> nn.Linear:
    """A linear layer made of rows ``start:stop`` of ``fused``; building it draws no random numbers."""
    part = nn.utils.skip_init(nn.Linear, fused.in_features, stop - start)
    with torch.no_grad():
        part.weight.copy_(fused.weight[start:stop])
        part.bias.copy_(fused.bias[start:stop])
    return part


class GatedAttention(nn.Module):
    """Multi-head attention whose output passes through a learned gate.

    Each head's output is multiplied elementwise by ``sigmoid(W x + b)`` of the
    query token, so a token can damp what it receives.

    With ``similarity_bias`` the block also accepts a prior per (query, key)
    pair, here the cosine between a candidate and a history item, which every
    head adds to its attention logits with its own learned scale. The scales
    start evenly spaced from 0 to 8, so the first head begins as plain
    attention and the last one close to a nearest-neighbour lookup in the
    content space.
    """

    def __init__(self, d: int, n_heads: int, dropout: float, gate_stream: int, similarity_bias: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.query = nn.Linear(d, d)
        # Keys and values are separate matrices because the optimiser
        # orthogonalises each hidden matrix on its own. Their initial values are
        # the two halves of one (2d x d) draw, as in the trained members.
        fused = nn.Linear(d, 2 * d)
        self.key = _split_linear(fused, 0, d)
        self.value = _split_linear(fused, d, 2 * d)
        self.output = nn.Linear(d, d)
        with side_stream(gate_stream):
            self.gate = nn.Linear(d, d)
        if similarity_bias:
            self.similarity_scale = nn.Parameter(torch.linspace(0.0, 8.0, n_heads))

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor, bias: Tensor | None = None) -> Tensor:
        """Attend from ``x`` [B, N, D] to ``context`` [B, K, D].

        ``context_mask`` [B, K] is True on real keys. ``bias`` [B, N, K] is the
        prior added to the logits; it is used only by blocks built with
        ``similarity_bias``.
        """
        B, N, D = x.shape
        K = context.shape[1]
        h = self.n_heads
        q = self.query(x).view(B, N, h, D // h).transpose(1, 2)
        k = self.key(context).view(B, K, h, D // h).transpose(1, 2)
        v = self.value(context).view(B, K, h, D // h).transpose(1, 2)
        if bias is not None and hasattr(self, "similarity_scale"):
            attention_mask = bias[:, None] * self.similarity_scale[None, :, None, None]
            attention_mask = attention_mask.masked_fill(~context_mask[:, None, None, :], float("-inf")).to(q.dtype)
        else:
            attention_mask = context_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        out = out * torch.sigmoid(self.gate(x)).view(B, N, h, D // h).transpose(1, 2)
        return self.output(out.transpose(1, 2).reshape(B, N, D))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attention, then a feed-forward layer, each with a residual.

    Called without ``context`` it is self-attention over ``x``; with ``context``
    it attends from ``x`` to that other sequence. ``norm_context`` is read only
    in the second case. The self-attention blocks carry it unused (2 d numbers
    each), which keeps the parameter count and the checkpoint layout of the
    trained members.
    """

    def __init__(self, d: int, n_heads: int, dropout: float, ffn_multiplier: int, gate_stream: int,
                 similarity_bias: bool = False):
        super().__init__()
        self.norm_query = nn.LayerNorm(d)
        self.norm_context = nn.LayerNorm(d)
        self.attention = GatedAttention(d, n_heads, dropout, gate_stream, similarity_bias)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_multiplier * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_multiplier * d, d))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context_mask: Tensor, context: Tensor | None = None,
                bias: Tensor | None = None) -> Tensor:
        query = self.norm_query(x)
        keys = query if context is None else self.norm_context(context)
        x = x + self.dropout(self.attention(query, keys, context_mask, bias))
        return x + self.dropout(self.ffn(self.norm_ffn(x)))


class HistoryRetrieval(nn.Module):
    """For each candidate, look up its most similar history items and summarise how they relate to it.

    Attention sees the history through 192-d tokens. This block gives the
    candidate a direct view of its nearest neighbours in the user's history
    (in the style of TabR):

    1. Take the ``n_slots`` history items with the highest cosine to the candidate.
    2. Project the candidate and those items to 64-d keys with one matrix.
    3. Weight each neighbour by a softmax over ``-|key distance|^2 + a * cosine
       + bias[event type]``. A learned "sink" logit competes in the softmax
       and carries no value, so a candidate with no relevant neighbour can
       retrieve nothing.
    4. A neighbour's value is its event type, event source and recency bucket
       plus an MLP of the key difference.
    5. The weighted sum of values goes through an output layer that starts at
       zero, so at initialisation the block changes nothing.
    """

    KEY_DIM = 64
    N_RECENCY_BUCKETS = 9

    def __init__(self, d: int, n_slots: int, dropout: float):
        super().__init__()
        self.n_slots = n_slots
        self.key = nn.Linear(EMBEDDING_DIM, self.KEY_DIM, bias=False)
        self.similarity_scale = nn.Parameter(torch.tensor(4.0))
        self.type_bias = nn.Embedding(N_EVENT_TYPES + 1, 1)
        nn.init.zeros_(self.type_bias.weight)
        self.sink = nn.Parameter(torch.zeros(1))
        self.event_type = nn.Embedding(N_EVENT_TYPES + 1, d)
        self.event_source = nn.Embedding(N_EVENT_SOURCES, d)
        self.recency = nn.Embedding(self.N_RECENCY_BUCKETS, d)
        self.difference_in = nn.Linear(self.KEY_DIM, 128)
        self.difference_out = nn.Linear(128, d, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.value_norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, d)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, candidate: Tensor, history: Tensor, history_has_embedding: Tensor, event_type: Tensor,
                event_source: Tensor, cosine: Tensor, candidate_no_embedding: Tensor) -> Tensor:
        """Return the retrieval summary ``[B, M, d]`` that is added to the candidate token.

        ``candidate`` [B, M, 128] and ``history`` [B, K, 128] are unit content
        embeddings and ``cosine`` [B, M, K] their cosine matrix.
        """
        B, M, _ = candidate.shape
        K = history.shape[1]
        n_slots = min(self.n_slots, K)
        top, slot = cosine.float().masked_fill(~history_has_embedding[:, None, :], -2.0).topk(n_slots, -1)
        valid = (top > -1.5) & ~candidate_no_embedding[..., None]

        candidate_key = self.key(candidate)
        history_key = self.key(history)
        flat = (slot + (torch.arange(B, device=candidate.device) * K)[:, None, None]).reshape(-1)
        neighbour_key = history_key.reshape(B * K, -1)[flat].view(B, M, n_slots, -1)
        difference = candidate_key[:, :, None, :] - neighbour_key
        neighbour_type = event_type.reshape(-1)[flat].view(B, M, n_slots)
        neighbour_source = event_source.reshape(-1)[flat].view(B, M, n_slots)
        # Recency bucket of the neighbour: floor(log2(1 + position)), position 0 = newest event.
        bucket = torch.floor(torch.log2(1.0 + slot.float())).long().clamp_max(self.N_RECENCY_BUCKETS - 1)

        with full_precision(candidate):
            logit = -(difference.float() ** 2).sum(-1) + self.similarity_scale * top \
                + self.type_bias(neighbour_type).squeeze(-1)
            logit = logit.masked_fill(~valid, float("-inf"))
            logit = torch.cat([self.sink.expand(B, M, 1), logit], -1)
            weight = torch.softmax(logit, -1)

        value = self.event_type(neighbour_type) + self.event_source(neighbour_source) + self.recency(bucket) \
            + self.difference_out(self.dropout(F.relu(self.difference_in(difference))))
        value = self.value_norm(value)
        summary = (weight[..., 1:, None].to(value.dtype) * value).sum(2)
        return self.output(summary)
