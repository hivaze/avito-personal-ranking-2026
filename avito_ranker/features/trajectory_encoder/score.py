"""Score the candidates of a set of feeds with a trained trajectory encoder.

For every feed the encoder reads the feed's own history snapshot and summarises
it in the state after the newest event. Six raw scores come out per candidate:

``fe``, ``new``   ``scale * <query_head(state), item vector>`` (+ item bias) for the two heads
``fe_noid``       the ``fe`` score with the candidate's ID embedding switched off (content and category only)
``hmax32``, ``hmax8``   the largest cosine, in the encoder's item space, between the candidate and the
                  32 (8) newest history items; -2 when the history is empty
``idmax32``       the same in the ID-embedding space; -2 when the candidate or all 32 history items have no ID

These are not yet ranker inputs: ``inputs.py`` aligns their scale across
models and averages over the two variants.

Which model may score which feed is the whole safety argument, see
``corpus.py``: val rows by the ``no_test`` models, test rows by the ``all``
models, train rows of a block by that block's ``window`` models.

Feeds are scored in chunks; inside a chunk they are sorted by history length
and batched. The forward pass runs under bf16 autocast, so the batch
composition changes the last bits of a score. The recipe keeps the chunk
sizes the stored scores were made with (30,000 feeds for val and test, 8,000
for the train blocks, 30,000 or 4,000 for the alignment samples); with them
the stored scores come out bit for bit.

Cost: loading a checkpoint (7 to 21 GB) and the item tables takes 20 to 60
seconds, scoring val or test a few seconds, a train block of 0.8M feeds about
half a minute. The process peaked at 63 GB of GPU memory while scoring a
train block.

Reproducibility. The stored checkpoints load into ``TrajectoryEncoder`` and
reproduce every stored score bit for bit with the recipe's chunk sizes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from avito_ranker.data.flatten import feed_offsets, read_feeds, read_flat, rows_of_feeds
from avito_ranker.features.trajectory_encoder.corpus import reverse_gather
from avito_ranker.features.trajectory_encoder.files import (
    SCORE_KEYS,
    checkpoint_file,
    corpus_dir,
    score_file,
)
from avito_ranker.features.trajectory_encoder.model import TrajectoryEncoder
from avito_ranker.features.trajectory_encoder.tables import ItemTables
from avito_ranker.features.trajectory_encoder.train import (
    HEADS,
    build_model,
    padded_index,
)
from avito_ranker.io import save_array
from avito_ranker.model.blocks import tf32_matmul

NEWEST_HISTORY = 32
TOKEN_BUDGET = 200_000  # padded tokens per scoring batch
MAX_HISTORY_WITH_START = 258
SAMPLE_SEED = 0


class FeedChunk:
    """A run of feeds on the device: histories oldest first, candidates, chunk-local offsets."""

    def __init__(self, device, n_hist, history: dict[str, np.ndarray], n_cand, cand_item):
        def on_device(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).to(device)

        self.n = len(n_hist)
        self.n_hist_np = n_hist
        self.item = on_device(history["item_id"])
        self.et = on_device(history["event_type"])
        self.es = on_device(history["event_source"])
        self.off = on_device(feed_offsets(n_hist))
        self.lens = on_device(n_hist.astype(np.int64))
        self.c_item = on_device(cand_item)
        self.c_feed = on_device(np.repeat(np.arange(self.n), n_cand).astype(np.int64))


@torch.no_grad()
def score_chunk(model: TrajectoryEncoder, items: ItemTables, chunk: FeedChunk) -> dict[str, np.ndarray]:
    """The six raw scores for every candidate of a chunk, float32."""
    model.eval()
    dev = chunk.item.device
    n = chunk.n
    queries = {head: torch.zeros((n, model.d), device=dev) for head in HEADS}
    hist_vec = torch.zeros((n, NEWEST_HISTORY, model.d), device=dev)  # newest history items in item space
    hist_ok = torch.zeros((n, NEWEST_HISTORY), dtype=torch.bool, device=dev)
    hist_id = torch.zeros((n, NEWEST_HISTORY, model.id_emb.weight.shape[1]), device=dev)
    order = np.argsort(chunk.n_hist_np, kind="stable")
    batch = max(1, TOKEN_BUDGET // MAX_HISTORY_WITH_START)
    for i in range(0, n, batch):
        ids = order[i:i + batch]
        S = torch.from_numpy(ids).to(dev)
        idx, mask, longest = padded_index(chunk.off, chunk.lens, S)
        if longest == 0:
            idx = torch.zeros((len(ids), 1), dtype=torch.long, device=dev)
            mask = torch.zeros_like(idx, dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            it = torch.where(mask, chunk.item[idx], 0)
            rep = items.rep(model, it, valid=mask)
            h = model.encode(rep, chunk.et[idx].long() * mask, chunk.es[idx].long() * mask)
        last = chunk.lens[S]  # the state after the newest event; 0 is the start token of an empty history
        rows = torch.arange(len(ids), device=dev)
        state = h[rows, last].float()
        for head in HEADS:
            queries[head][S] = model.query(state, head)
        back = torch.arange(NEWEST_HISTORY, device=dev)
        pos = last[:, None] - 1 - back[None, :]
        present = pos >= 0
        hist_vec[S] = model.out_vec(rep.float()[rows[:, None], pos.clamp_min(0)]) * present[..., None]
        hist_ok[S] = present
        newest_items = torch.where(mask, chunk.item[idx], 0).long()[rows[:, None], pos.clamp_min(0)]
        slot = items.slot[newest_items].long() * present
        hist_id[S] = F.normalize(model.id_emb(slot).float(), dim=-1) * (slot > 0)[..., None]

    scale = model.logit_scale.exp().clamp(max=100.0)
    n_cand = len(chunk.c_item)
    vec = torch.zeros((n_cand, model.d), device=dev)
    vec_noid = torch.zeros((n_cand, model.d), device=dev)
    for a in range(0, n_cand, 500_000):
        it = chunk.c_item[a:a + 500_000]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = items.rep(model, it)
            itl = it.long()
            r_noid = model.item_rep(items.tab[itl], items.has[itl], items.cat[itl].long(), torch.zeros_like(itl))
        vec[a:a + 500_000] = model.out_vec(r.float())
        vec_noid[a:a + 500_000] = model.out_vec(r_noid.float())
    bias = model.bias(items.slot[chunk.c_item.long()]) if model.id_bias is not None else torch.zeros(n_cand, device=dev)
    out = {}
    for head in HEADS:
        q = queries[head][chunk.c_feed]
        out[head] = (scale * (q * vec).sum(-1) + bias).cpu().numpy().astype(np.float32)
        if head == "fe":
            out["fe_noid"] = (scale * (q * vec_noid).sum(-1)).cpu().numpy().astype(np.float32)

    sims = []
    for a in range(0, n_cand, 200_000):
        feed = chunk.c_feed[a:a + 200_000]
        s = torch.einsum("nd,nkd->nk", vec[a:a + 200_000], hist_vec[feed])
        sims.append(s.masked_fill(~hist_ok[feed], -2.0))
    s = torch.cat(sims)
    out["hmax32"] = s.max(1).values.cpu().numpy().astype(np.float32)
    out["hmax8"] = s[:, :8].max(1).values.cpu().numpy().astype(np.float32)

    sims = []
    cand_slot = items.slot[chunk.c_item.long()].long()
    cand_id = F.normalize(model.id_emb(cand_slot).float(), dim=-1)
    for a in range(0, n_cand, 200_000):
        ids_of_history = hist_id[chunk.c_feed[a:a + 200_000]]
        s = torch.einsum("nd,nkd->nk", cand_id[a:a + 200_000], ids_of_history)
        s = s.masked_fill((ids_of_history.abs().sum(-1) == 0) | (cand_slot[a:a + 200_000] == 0)[:, None], -2.0)
        sims.append(s.max(1).values)
    out["idmax32"] = torch.cat(sims).cpu().numpy().astype(np.float32)
    model.train()
    return out


def load_encoder(model_name: str, content: str, recipe: dict,
                 device: torch.device) -> tuple[TrajectoryEncoder, ItemTables]:
    """A stored model (``<corpus>_<variant>``) with the item tables of its corpus and the chosen content table."""
    corpus, variant = split_model_name(model_name)
    items = ItemTables(device, corpus, recipe["model"]["min_item_count"], content)
    model = build_model(items.n_id, recipe["model"], recipe["variants"][variant])
    state = torch.load(checkpoint_file(model_name), map_location="cpu")["model"]
    model.load_state_dict(state)
    return model.to(device), items


def split_model_name(model_name: str) -> tuple[str, str]:
    """``"window1_id128_bias"`` -> ``("window1", "id128_bias")``."""
    for corpus in ("no_test", "all", "window1", "window2", "window3"):
        if model_name.startswith(corpus + "_"):
            return corpus, model_name[len(corpus) + 1:]
    raise ValueError(f"unknown encoder model {model_name!r}")


def target_feeds(target: str, sample_feeds: int) -> tuple[str, np.ndarray]:
    """``(split, feed indices)`` of a scoring target.

    ``val`` and ``test`` are whole splits; ``window<k>`` is every train feed
    of the block; ``window<k>_sample`` a fixed random sample of them, on which
    the block's model and the ``all`` model are compared to align their scales.
    """
    if target in ("val", "test"):
        return target, np.arange(len(read_feeds(target)))
    window, _, sample = target.partition("_")
    feeds = np.load(corpus_dir(window) / "window_feeds.npy")
    if sample:
        rng = np.random.default_rng(SAMPLE_SEED)
        feeds = np.sort(rng.choice(feeds, min(sample_feeds, len(feeds)), replace=False))
    return "train", feeds


def candidate_rows(split: str, feeds: np.ndarray) -> np.ndarray:
    """Flat candidate rows of the given feeds of a split, feed after feed."""
    table = read_feeds(split)
    first_rows = table["cand_off"].to_numpy().astype(np.int64)[feeds]
    return rows_of_feeds(first_rows, table["M"].to_numpy().astype(np.int64)[feeds])


def score_feeds(model: TrajectoryEncoder, items: ItemTables, split: str, feeds: np.ndarray,
                chunk_feeds: int) -> dict[str, np.ndarray]:
    """Raw scores for the candidates of ``feeds``, in the order of ``candidate_rows(split, feeds)``.

    Float32 products use TF32, as in the runs that wrote the stored scores.
    """
    with tf32_matmul(True, cudnn=True):
        return _score_feeds(model, items, split, feeds, chunk_feeds)


def _score_feeds(model: TrajectoryEncoder, items: ItemTables, split: str, feeds: np.ndarray,
                 chunk_feeds: int) -> dict[str, np.ndarray]:
    """The body of :func:`score_feeds`."""
    device = items.tab.device
    table = read_feeds(split)
    n_hist = table["H"].to_numpy().astype(np.int64)[feeds]
    hist_off = table["hist_off"].to_numpy().astype(np.int64)[feeds]
    n_cand = table["M"].to_numpy().astype(np.int64)[feeds]
    out_off = feed_offsets(n_hist)
    history = {}
    for field in ("item_id", "event_type", "event_source"):
        source = np.ascontiguousarray(read_flat(split, f"hist_{field}", mmap=split != "train"))
        history[field] = np.empty(int(n_hist.sum()), source.dtype)
        reverse_gather(hist_off, n_hist, source, out_off, history[field])
        del source
    cand_item = read_flat(split, "cand_item_id")[candidate_rows(split, feeds)]
    cand_end = np.concatenate([[0], np.cumsum(n_cand)]).astype(np.int64)
    hist_end = np.concatenate([[0], np.cumsum(n_hist)]).astype(np.int64)

    out = {key: np.empty(cand_end[-1], np.float32) for key in SCORE_KEYS}
    for i in range(0, len(feeds), chunk_feeds):
        j = min(i + chunk_feeds, len(feeds))
        hs, he, cs, ce = hist_end[i], hist_end[j], cand_end[i], cand_end[j]
        chunk = FeedChunk(device, n_hist[i:j], {k: v[hs:he] for k, v in history.items()}, n_cand[i:j], cand_item[cs:ce])
        for key, values in score_chunk(model, items, chunk).items():
            out[key][cs:ce] = values
        del chunk
    return out


def save_scores(content: str, target: str, model_name: str, scores: dict[str, np.ndarray],
                overwrite: bool = False) -> None:
    """Write the six raw scores of a model on a target.

    Stored files with exactly these values are left alone; other values are
    replaced only when ``overwrite`` is set.
    """
    for key in SCORE_KEYS:
        save_array(score_file(content, target, model_name, key), scores[key], overwrite)
