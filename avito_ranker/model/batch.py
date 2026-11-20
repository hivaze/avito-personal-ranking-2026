"""The batch a ranker consumes: a set of feeds padded to common lengths.

``B`` feeds, each padded to ``M`` candidates and ``K`` history events. The
history is stored newest first, so column 0 of every history tensor is the
user's most recent event. ``avito_ranker.data.loader`` builds batches from the
flat arrays of a split; tests build small ones by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def unit_length(vectors: Tensor) -> Tensor:
    """Scale vectors along the last axis to length 1. All-zero vectors (missing embeddings) stay zero."""
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-3)


@dataclass
class Batch:
    """Padded tensors of ``B`` feeds on one device.

    Padded candidate slots repeat the feed's first candidate. Padded history
    slots have a zero embedding and arbitrary codes. ``candidate_mask`` and
    ``history_mask`` say which slots are real.
    """

    feed_index: Tensor               # [B]        row of each feed in the split's feed table
    feed_size: Tensor                # [B]        number of candidates of the feed
    history_length: Tensor           # [B]        number of history events kept
    is_anonymous: Tensor             # [B]        feed of a not-logged-in session (user_id 0)
    candidate_mask: Tensor           # [B, M]     True on real candidates
    candidate_row: Tensor            # [B, M]     row of the candidate in the split's flat arrays
    candidate_embedding: Tensor      # [B, M, 128] content embedding as stored (int8 values), 0 if missing
    candidate_unit: Tensor           # [B, M, 128] the same at unit length
    candidate_no_embedding: Tensor   # [B, M]     True where the item has no content embedding
    candidate_category: Tensor       # [B, M]
    candidate_numeric: Tensor        # [B, M, 5]  the organisers' numeric features f1..f5
    history_mask: Tensor             # [B, K]     True on real events
    history_unit: Tensor             # [B, K, 128] content embedding of the event's item at unit length
    history_has_embedding: Tensor    # [B, K]     real event whose item has a content embedding
    history_category: Tensor         # [B, K]
    history_event_type: Tensor       # [B, K]     1..10
    history_event_source: Tensor     # [B, K]
    user_embedding: Tensor           # [B, 128]   user embedding as stored (int8 values)
    user_unit: Tensor                # [B, 128]   the same at unit length
    inputs: Tensor                   # [B, M, n_inputs] pre-computed inputs of the view, float32
    target: Tensor | None = None     # [B, M]     click label, 0 on padding (train and val)
    position: Tensor | None = None   # [B, M]     position in the production order (train only)
    domain_mix: Tensor | None = None       # [B, 3]  weights of the three traffic domains (train only)
    finetune_extras: Tensor | None = None  # [B, M, n_extras] inputs of the fine-tune branch (val and test)

    @property
    def device(self) -> torch.device:
        """The device that holds the batch."""
        return self.candidate_mask.device
