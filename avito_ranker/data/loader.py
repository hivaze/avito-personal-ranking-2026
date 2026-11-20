"""A split on the GPU, and the batches cut from it.

There is no CPU data loader. Every array of a split (the flat candidate and
history arrays, the int8 item-embedding table, the pre-computed inputs) is
copied to the GPU once; a batch is then a list of feed numbers, and
:meth:`SplitTensors.gather` builds its padded tensors with a few indexing
operations on the device. GPU memory of the train split, measured on
2026-11-16 on one GPU with the arrays in the page cache:

  view              inputs  after loading  peak while loading  load time
  full                  34       36.8 GiB            52.5 GiB       36 s
  popularity-light      27       33.6 GiB            46.0 GiB       29 s

Of the 36.8 GiB, the item table takes 9.7 GiB and the inputs in float16
15.6 GiB. Val and test take under 1 GiB each.

Feeds have 5 to 300 candidates. To waste little on padding, feeds are bucketed
by size rounded up to a multiple of 16 and a batch holds feeds of one bucket:
at most 512 feeds and at most 32,768 padded candidates in training.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from avito_ranker.data.flatten import has_flat, read_feeds, read_flat, read_item_embeddings
from avito_ranker.features.store import load_input
from avito_ranker.model.batch import Batch, unit_length
from avito_ranker.model.config import MAX_HISTORY, load_recipe
from avito_ranker.model.inputs import InputView

SIZE_STEP = 16      # candidate slots are padded to a multiple of this
HISTORY_STEP = 64   # history slots are padded to a multiple of this


@dataclass(frozen=True)
class ExtrasScaling:
    """How the fine-tune inputs are standardised: fitted on val, reused for test."""

    log_columns: np.ndarray  # bool per input: count-like, passed through log1p first
    mean: np.ndarray
    std: np.ndarray


@dataclass
class SplitTensors:
    """All tensors of one split on one device. Rows are in the order of the flat files."""

    name: str
    view: InputView
    device: torch.device
    max_history: int

    # Per feed, on the host: what batch planning and evaluation need.
    feed_sizes: np.ndarray         # int64, candidates per feed
    history_lengths: np.ndarray    # int64, history events per feed
    candidate_offsets: np.ndarray  # int64, first candidate row of each feed
    user_ids: np.ndarray           # 0 = not-logged-in session
    targets: np.ndarray | None     # float32 click label per candidate row; None for test

    # On the device.
    feed_size: Tensor
    history_length: Tensor
    candidate_offset: Tensor
    history_offset: Tensor
    user_embedding: Tensor               # int8 [n_feeds, 128]
    is_anonymous: Tensor                 # bool [n_feeds]
    candidate_embedding_row: Tensor      # int32, row of the item table, -1 = no embedding
    candidate_category: Tensor           # int16
    candidate_numeric: Tensor            # float32 [n_candidates, 5]: f1..f5
    candidate_target: Tensor | None      # int8
    candidate_position: Tensor | None    # int16, production position (train only)
    history_embedding_row: Tensor        # int32
    history_category: Tensor             # int16
    history_event_type: Tensor           # int8, 1..10
    history_event_source: Tensor         # int8
    item_table: Tensor                   # int8 [n_items, 128], this split's content embeddings
    inputs: Tensor                       # float16 [n_candidates, n_inputs], the view's inputs after scaling

    # Set by the stages that need them.
    domain_mix: Tensor | None = None       # float32 [n_feeds, 3], see set_domain_mix
    finetune_extras: Tensor | None = None  # float32 [n_candidates, n_extras], see attach_finetune_extras

    @property
    def n_feeds(self) -> int:
        """Number of feeds of the split."""
        return len(self.feed_sizes)

    @property
    def n_candidates(self) -> int:
        """Number of candidate rows of the split."""
        return len(self.candidate_embedding_row)

    @property
    def n_history(self) -> int:
        """Number of history events of the split."""
        return len(self.history_embedding_row)

    def summary(self) -> str:
        """One line for a log: sizes of the split."""
        return (f"{self.name}: {self.n_feeds:,} feeds, {self.n_candidates:,} candidates, "
                f"{self.n_history:,} history events, {self.item_table.shape[0]:,} item embeddings, "
                f"{self.view.n_inputs} inputs")

    def set_domain_mix(self, random_order_probability: np.ndarray) -> None:
        """Give every feed its weights over the model's three domain rows.

        Used for the train split: a feed is production traffic with weight
        ``1 - p`` and random-order traffic with weight ``p``, where ``p`` is the
        probability that it was shown in random order. The third row belongs to
        the fine-tune and gets 0.
        """
        p = torch.from_numpy(np.asarray(random_order_probability, dtype=np.float32)).to(self.device)
        self.domain_mix = torch.stack([1 - p, p, torch.zeros_like(p)], 1)

    def attach_finetune_extras(self, files: Sequence[str], scaling: ExtrasScaling | None = None) -> ExtrasScaling:
        """Load the inputs of the fine-tune branch and standardise them.

        Count-like inputs (never negative, maximum above 20) go through
        ``log1p``; every input is then standardised, clipped to [-6, 6] and its
        missing values set to 0. Call it on val without ``scaling`` and pass
        the returned scaling to test, so both splits are treated alike.
        """
        columns = [load_input(self.name, file, mmap=False).astype(np.float32) for file in files]
        values = np.stack(columns, 1)
        log_columns = scaling.log_columns if scaling else (np.nanmin(values, 0) >= 0) & (np.nanmax(values, 0) > 20)
        values[:, log_columns] = np.log1p(np.maximum(values[:, log_columns], 0))
        if scaling is None:
            scaling = ExtrasScaling(log_columns, np.nanmean(values, 0), np.nanstd(values, 0) + 1e-6)
        values = np.clip((values - scaling.mean) / scaling.std, -6, 6)
        values = np.nan_to_num(values, nan=0.0)
        self.finetune_extras = torch.from_numpy(values.astype(np.float32)).to(self.device)
        return scaling

    def plan_batches(self, feed_ids: np.ndarray, max_candidates: int = 32768, max_feeds: int = 512,
                     shuffle: bool = False, rng: np.random.Generator | None = None) -> list[np.ndarray]:
        """Cut ``feed_ids`` of this split into batches of feeds of similar size (see :func:`plan_batches`)."""
        return plan_batches(self.feed_sizes, feed_ids, max_candidates, max_feeds, shuffle, rng)

    @torch.no_grad()
    def gather(self, feed_ids: np.ndarray, with_positions: bool = False) -> Batch:
        """Build the padded tensors of the given feeds.

        Candidates are padded to the largest feed of the batch rounded up to a
        multiple of 16, the history to the longest one rounded up to a multiple
        of 64 (at most ``max_history`` events, the newest ones).
        """
        device = self.device
        feed_ids = np.asarray(feed_ids)
        feed = torch.from_numpy(feed_ids.astype(np.int64)).to(device, non_blocking=True)

        size = self.feed_size[feed]
        width = int(-(-self.feed_sizes[feed_ids].max() // SIZE_STEP) * SIZE_STEP)
        slot = torch.arange(width, device=device)
        candidate_mask = slot[None, :] < size[:, None]
        first_row = self.candidate_offset[feed][:, None]
        row = torch.where(candidate_mask, first_row + slot[None, :], first_row)
        embedding_row = self.candidate_embedding_row[row]
        no_embedding = embedding_row < 0
        embedding = self.item_table[embedding_row.clamp_min(0).long()].float()
        embedding[no_embedding] = 0.0

        history_length = self.history_length[feed].clamp_max(self.max_history)
        depth = int(np.minimum(self.history_lengths[feed_ids], self.max_history).max())
        depth = min(max(-(-depth // HISTORY_STEP) * HISTORY_STEP, HISTORY_STEP), self.max_history)
        history_slot = torch.arange(depth, device=device)
        history_mask = history_slot[None, :] < history_length[:, None]
        first_event = self.history_offset[feed][:, None]
        event = torch.where(history_mask, first_event + history_slot[None, :], first_event)
        event = event.clamp_max(self.n_history - 1)
        history_embedding_row = self.history_embedding_row[event]
        history_missing = (history_embedding_row < 0) | ~history_mask
        history_embedding = self.item_table[history_embedding_row.clamp_min(0).long()].float()
        history_embedding[history_missing] = 0.0

        user_embedding = self.user_embedding[feed].float()
        batch = Batch(
            feed_index=feed,
            feed_size=size,
            history_length=history_length,
            is_anonymous=self.is_anonymous[feed],
            candidate_mask=candidate_mask,
            candidate_row=row,
            candidate_embedding=embedding,
            candidate_unit=unit_length(embedding),
            candidate_no_embedding=no_embedding,
            candidate_category=self.candidate_category[row].long(),
            candidate_numeric=self.candidate_numeric[row],
            history_mask=history_mask,
            history_unit=unit_length(history_embedding),
            history_has_embedding=~history_missing,
            history_category=self.history_category[event].long(),
            history_event_type=self.history_event_type[event].long(),
            history_event_source=self.history_event_source[event].long(),
            user_embedding=user_embedding,
            user_unit=unit_length(user_embedding),
            inputs=self.inputs[row].float(),
        )
        if self.candidate_target is not None:
            batch.target = self.candidate_target[row].float() * candidate_mask
        if self.domain_mix is not None:
            batch.domain_mix = self.domain_mix[feed]
        if self.finetune_extras is not None:
            batch.finetune_extras = self.finetune_extras[row]
        if with_positions and self.candidate_position is not None:
            batch.position = self.candidate_position[row].long()
        return batch

    def scatter(self, flat_scores: Tensor, batch: Batch, scores: Tensor) -> None:
        """Write the padded scores ``[B, M]`` of a batch into a flat buffer aligned with the candidate rows."""
        mask = batch.candidate_mask
        flat_scores[batch.candidate_row[mask]] = scores[mask].float()


def plan_batches(feed_sizes: np.ndarray, feed_ids: np.ndarray, max_candidates: int = 32768, max_feeds: int = 512,
                 shuffle: bool = False, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Cut ``feed_ids`` into batches of feeds of similar size; ``feed_sizes`` holds every feed's size.

    Feeds are bucketed by size rounded up to a multiple of 16. A bucket of
    padded size ``m`` is cut into batches of ``min(max_feeds,
    max_candidates // m)`` feeds. With ``shuffle`` the feeds inside each
    bucket and then the batches are permuted with ``rng``, which is then
    required. The number of batches does not depend on the shuffle.

    Few distinct batch shapes matter for speed: every new shape costs the
    GPU about a quarter of a second of kernel selection.
    """
    if shuffle and rng is None:
        raise ValueError("shuffle needs a random generator (rng)")
    feed_ids = np.asarray(feed_ids)
    padded = (feed_sizes[feed_ids] + SIZE_STEP - 1) // SIZE_STEP * SIZE_STEP
    batches = []
    for size in np.unique(padded):
        bucket = feed_ids[padded == size]
        if shuffle:
            bucket = rng.permutation(bucket)
        per_batch = int(max(1, min(max_feeds, max_candidates // size)))
        batches += [bucket[i:i + per_batch] for i in range(0, len(bucket), per_batch)]
    if shuffle:
        order = rng.permutation(len(batches))
        batches = [batches[k] for k in order]
    return batches


def _inputs_on_device(split: str, view: InputView, device: torch.device) -> Tensor:
    """The view's inputs after their fixed scaling, float16 ``[n_candidates, n_inputs]``.

    Columns are moved to the device one at a time: the float32 matrix of the
    train split (246.7M rows x 34) would need 33 GB of host memory.
    """
    columns = [torch.from_numpy(column.scale(load_input(split, column.file, mmap=False))).to(device, torch.float16)
               for column in view.columns]
    return torch.stack(columns, 1)


def load_split(split: str, view: InputView | str, device: str | torch.device = "cuda",
               max_history: int = MAX_HISTORY) -> SplitTensors:
    """Copy a split (``"train"``, ``"val"`` or ``"test"``) to ``device``.

    ``view`` is the list of pre-computed inputs to attach: an
    :class:`~avito_ranker.model.inputs.InputView` (``config.view``) or the name
    of a model recipe such as ``"full"`` or ``"popularity_light"``.
    """
    if isinstance(view, str):
        view = load_recipe(view).view
    device = torch.device(device)
    feeds = read_feeds(split)

    def on_device(array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(device)

    def flat(name: str) -> np.ndarray:
        return read_flat(split, name, mmap=False)

    feed_sizes = feeds["M"].to_numpy().astype(np.int64)
    history_lengths = feeds["H"].to_numpy().astype(np.int64)
    candidate_offsets = feeds["cand_off"].to_numpy().astype(np.int64)
    user_ids = feeds["user_id"].to_numpy()
    targets = flat("cand_target").astype(np.float32) if has_flat(split, "cand_target") else None

    return SplitTensors(
        name=split,
        view=view,
        device=device,
        max_history=max_history,
        feed_sizes=feed_sizes,
        history_lengths=history_lengths,
        candidate_offsets=candidate_offsets,
        user_ids=user_ids,
        targets=targets,
        feed_size=on_device(feed_sizes),
        history_length=on_device(history_lengths),
        candidate_offset=on_device(candidate_offsets),
        history_offset=on_device(feeds["hist_off"].to_numpy().astype(np.int64)),
        user_embedding=on_device(flat("user_emb")),
        is_anonymous=on_device(user_ids == 0),
        candidate_embedding_row=on_device(flat("cand_emb_idx")),
        candidate_category=on_device(flat("cand_category_id").astype(np.int16)),
        candidate_numeric=torch.stack([on_device(flat(f"cand_f{k}")) for k in range(1, 6)], dim=1),
        candidate_target=None if targets is None else on_device(targets.astype(np.int8)),
        candidate_position=on_device(flat("cand_pos")) if has_flat(split, "cand_pos") else None,
        history_embedding_row=on_device(flat("hist_emb_idx")),
        history_category=on_device(flat("hist_category_id").astype(np.int16)),
        history_event_type=on_device(flat("hist_event_type")),
        history_event_source=on_device(flat("hist_event_source")),
        item_table=on_device(np.array(read_item_embeddings(split))),
        inputs=_inputs_on_device(split, view, device),
    )
