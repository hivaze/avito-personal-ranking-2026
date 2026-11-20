"""The trajectory encoder: a causal transformer over a user's history events.

The encoder reads the history oldest to newest. Each event becomes a token

    token = item_tower(item) + E_event_type + E_event_source

and a learned start token is put in front, so that an empty history still has
a state. Four pre-LN transformer blocks with causal attention and rotary
position encoding turn the tokens into states. The state after the newest
event is the summary of the user at the time of the feed.

The item tower is shared by history items and candidates:

    item_tower(i) = MLP(unit content embedding of i)   (a learned vector when i has none)
                    + E_category
                    + P * E_id[slot(i)]                 (only for items seen at least 3 times in the corpus)

During training the ID part is dropped at random for 30% of the items, so the
content path has to work alone; that is what makes scores for cold candidates
meaningful.

Two linear heads turn a state into a query, one per self-supervised target:
``fe`` (the next new item the user opens from a feed) and ``new`` (the next
new item of any kind). The score of an item for a state is

    scale * <normalize(head(state)), normalize(W_out item_tower(item))>   (+ a per-item bias in one variant)

Attribute names are kept from the first implementation, so the stored
checkpoints load without a key map.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from avito_ranker.data.flatten import N_CATEGORIES, N_EVENT_SOURCES, N_EVENT_TYPES

# Rows of the embedding tables. Event types are coded 1..10 and a padded position carries 0, so
# the event-type table has 11 rows (the data's count is avito_ranker.data.flatten.N_EVENT_TYPES).
EVENT_TYPE_ROWS = N_EVENT_TYPES + 1
EVENT_SOURCE_ROWS = N_EVENT_SOURCES
CATEGORY_ROWS = N_CATEGORIES
CONTENT_DIM = 128


def rotary_tables(length: int, head_dim: int, device: str | torch.device,
                  base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables of the rotary position encoding, ``[length, head_dim / 2]`` each."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angle = torch.outer(torch.arange(length, device=device).float(), inv)
    return angle.cos(), angle.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels of ``x [B, heads, L, head_dim]`` by their position."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None, :x.shape[2]], sin[None, None, :x.shape[2]]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Block(nn.Module):
    """Pre-LN transformer block with causal self-attention."""

    def __init__(self, d: int, nh: int, drop: float):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)
        self.pdrop = drop

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.nh, d // self.nh).permute(2, 0, 3, 1, 4)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.pdrop if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class TrajectoryEncoder(nn.Module):
    """Item tower, causal encoder and the two query heads.

    ``n_id`` is the number of items that own a row of the ID table; row 0 is
    the padding row for items without an ID.
    """

    def __init__(self, n_id: int, d: int = 384, nl: int = 4, nh: int = 6, d_id: int = 64, drop: float = 0.05,
                 heads: tuple[str, ...] = ("fe", "new"), max_len: int = 320, id_bias: bool = False):
        super().__init__()
        self.d = d
        self.content = nn.Sequential(nn.Linear(CONTENT_DIM, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.missing = nn.Parameter(torch.zeros(d))
        self.cat = nn.Embedding(CATEGORY_ROWS, d)
        self.n_id = n_id
        self.id_emb = nn.Embedding(n_id + 1, d_id, sparse=True, padding_idx=0)
        nn.init.normal_(self.id_emb.weight, std=0.02)
        with torch.no_grad():
            self.id_emb.weight[0].zero_()
        self.id_proj = nn.Linear(d_id, d, bias=False)
        self.id_bias = nn.Embedding(n_id + 1, 1, sparse=True, padding_idx=0) if id_bias else None
        if self.id_bias is not None:
            nn.init.zeros_(self.id_bias.weight)
        self.et = nn.Embedding(EVENT_TYPE_ROWS, d)
        self.es = nn.Embedding(EVENT_SOURCE_ROWS, d)
        self.bos = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, nh, drop) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.heads = nn.ModuleDict({k: nn.Linear(d, d) for k in heads})
        self.out_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        cos, sin = rotary_tables(max_len, d // nh, "cpu")
        self.register_buffer("rcos", cos, persistent=False)
        self.register_buffer("rsin", sin, persistent=False)

    def item_rep(self, emb_i8: torch.Tensor, has: torch.Tensor, cat: torch.Tensor, slot: torch.Tensor,
                 id_drop: float = 0.0) -> torch.Tensor:
        """Item-tower output.

        ``emb_i8`` int8 ``[..., 128]``, ``has`` bool, ``cat`` and ``slot`` long (slot 0 = no ID).
        """
        e = emb_i8.float()
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        r = self.content(e)
        r = torch.where(has[..., None], r, self.missing.to(r.dtype))
        r = r + self.cat(cat)
        if id_drop > 0 and self.training:
            slot = torch.where(torch.rand(slot.shape, device=slot.device) < id_drop, torch.zeros_like(slot), slot)
        return r + self.id_proj(self.id_emb(slot))

    def bias(self, slot: torch.Tensor) -> torch.Tensor:
        """Per-item logit bias (zeros in the variant without one)."""
        if self.id_bias is None:
            return torch.zeros(slot.shape, device=slot.device)
        return self.id_bias(slot.long()).squeeze(-1).float()

    def out_vec(self, rep: torch.Tensor) -> torch.Tensor:
        """Unit vector of an item in the space where queries and items are compared."""
        return F.normalize(self.out_proj(rep).float(), dim=-1)

    def encode(self, rep: torch.Tensor, et: torch.Tensor, es: torch.Tensor) -> torch.Tensor:
        """States of a right-padded batch of sequences.

        ``rep [B, L, d]`` are item-tower outputs, ``et`` and ``es`` long
        ``[B, L]``. Returns ``[B, L + 1, d]``: position 0 is the start token,
        position ``t + 1`` the state after event ``t``.
        """
        x = rep + self.et(et) + self.es(es)
        B = x.shape[0]
        x = torch.cat([self.bos.to(x.dtype).expand(B, 1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x, self.rcos, self.rsin)
        return self.lnf(x)

    def query(self, h: torch.Tensor, head: str) -> torch.Tensor:
        """Unit query vector of a state for one of the heads."""
        return F.normalize(self.heads[head](h).float(), dim=-1)
