"""Per-item lookup tables of the trajectory encoder, held on the GPU.

Training and scoring both turn item ids into item-tower inputs millions of
times per second, so everything the tower needs is indexed by item id and
kept on the device: the content embedding, whether the item has one, its
category, and its row in the model's ID table.
"""
from __future__ import annotations

import numpy as np
import torch

from avito_ranker.data.flatten import SPLITS, read_flat
from avito_ranker.features.trajectory_encoder.files import content_files, corpus_dir
from avito_ranker.features.trajectory_encoder.model import TrajectoryEncoder

COPY_STEP_ITEMS = 8_000_000  # the 14 GB content table goes to the GPU in pieces


def corpus_counts(corpus: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per item: occurrences as a token, as a ``fe`` target and as a ``new`` target in a corpus.

    For ``no_test`` the test sequences of the ``all`` corpus are left out, so
    an item known only from test snapshots gets no ID row.
    """
    folder = corpus_dir(corpus)
    if corpus != "no_test":
        return tuple(np.load(folder / f"{name}.npy") for name in ("item_cnt", "item_cnt_fe", "item_cnt_new"))
    seq_off = np.load(folder / "seq_off.npy")
    keep = np.repeat(np.load(folder / "seq_split.npy") != SPLITS.index("test"), np.diff(seq_off))
    n_items = len(np.load(content_files("partial")[1], mmap_mode="r"))
    token = np.load(folder / "tok_item.npy")[keep]
    fe = np.load(folder / "tgt_fe.npy")[keep]
    new = np.load(folder / "tgt_new.npy")[keep]
    return (np.bincount(token, minlength=n_items).astype(np.int32),
            np.bincount(fe[fe >= 0], minlength=n_items).astype(np.int32),
            np.bincount(new[new >= 0], minlength=n_items).astype(np.int32))


class ItemTables:
    """Lookup tables indexed by item id, for one model's corpus and one content table."""

    def __init__(self, device: torch.device, corpus: str, min_count: int, content: str):
        table_file, mask_file = content_files(content)
        table = np.load(table_file, mmap_mode="r")
        self.n_items = table.shape[0]
        self.tab = torch.empty((self.n_items, 128), dtype=torch.int8, device=device)
        for a in range(0, self.n_items, COPY_STEP_ITEMS):
            self.tab[a:a + COPY_STEP_ITEMS] = torch.from_numpy(np.array(table[a:a + COPY_STEP_ITEMS])).to(device)
        self.has = torch.from_numpy(np.load(mask_file)).to(device)

        count, self.count_fe, self.count_new = corpus_counts(corpus)
        slot = np.zeros(self.n_items, np.int32)
        with_id = np.where(count >= min_count)[0]
        slot[with_id] = np.arange(1, len(with_id) + 1, dtype=np.int32)
        self.n_id = len(with_id)
        self.slot = torch.from_numpy(slot).to(device)

        folder = corpus_dir(corpus)
        category = np.zeros(self.n_items, np.int8)
        category[np.load(folder / "tok_item.npy")] = np.load(folder / "tok_cat.npy")
        for split in SPLITS:
            category[read_flat(split, "cand_item_id", mmap=False)] = read_flat(split, "cand_category_id", mmap=False)
        self.cat = torch.from_numpy(category).to(device)
        self.corpus_items = torch.from_numpy(np.where(count > 0)[0].astype(np.int32)).to(device)
        self.logq = None

    def set_sampling_correction(self, n_fe: int, n_new: int, n_uniform: int) -> None:
        """Log of each item's expected count among the sampled negatives (the logQ correction)."""
        q = (n_fe * self.count_fe / self.count_fe.sum() + n_new * self.count_new / self.count_new.sum()
             + n_uniform / len(self.corpus_items)).astype(np.float32)
        self.logq = torch.from_numpy(np.log(np.maximum(q, 1e-12))).to(self.tab.device)

    def rep(self, model: TrajectoryEncoder, items: torch.Tensor, valid: torch.Tensor | None = None,
            id_drop: float = 0.0) -> torch.Tensor:
        """Item-tower output for a tensor of item ids; ``valid`` masks padding positions."""
        it = items.long().clamp_min(0)
        has = self.has[it]
        if valid is not None:
            has = has & valid
        return model.item_rep(self.tab[it], has, self.cat[it].long(), self.slot[it].long(), id_drop)
