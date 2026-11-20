"""Self-supervised training of the trajectory encoder.

Every position of every corpus sequence is a training example: from the state
after event ``t`` the encoder has to pick the user's next new item among
sampled negatives. Two losses share the encoder, one per head:

``fe``   next new item opened from a feed (one of the next four, drawn at random each step)
``new``  next new item of any kind, weight 0.5

Each loss is a sampled softmax over the positive and one shared pool of
negatives: 6,144 ``fe`` targets of the batch, 2,048 ``new`` targets of the
batch and 8,192 items drawn uniformly from the corpus. All logits get the
logQ correction, so popular items are not punished for being sampled often,
and a negative that equals the positive is masked.

Sequences are bucketed by length (steps of 32) and a batch holds as many
sequences as fit a budget of padded tokens. The dense weights use AdamW, the
ID table sparse Adam with a ten times larger learning rate; both follow a
300-step warm-up and a cosine decay to 10%. The forward pass runs under bf16
autocast.

The content table seen during training is set per corpus in the recipe: the
``all`` and ``no_test`` models were trained with the partial table, the window
models with the full one. Training reads only items of the corpus, and the two
tables differ on 16 of its 92.3M items, so the choice matters little; it is
kept as it was.

Cost: 21 to 43 minutes per ``all`` or ``no_test`` model (7,866 and 7,544
steps), 13 to 59 minutes per window model (4,598 to 7,944 steps), 5.2
GPU-hours for the ten (from the original training logs).

Reproducibility. Training on the GPU is not run-to-run reproducible; the
stored checkpoints, not a re-training, reproduce the stored scores (see
docs/10-reproduce.md).
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import numba as nb
import numpy as np
import torch

from avito_ranker.data.flatten import SPLITS
from avito_ranker.features.trajectory_encoder.files import checkpoint_file, corpus_dir
from avito_ranker.features.trajectory_encoder.model import TrajectoryEncoder
from avito_ranker.features.trajectory_encoder.tables import ItemTables
from avito_ranker.model.blocks import tf32_matmul

HEADS = ("fe", "new")
LENGTH_BUCKET = 32
LOG_EVERY = 200


@nb.njit(cache=True)
def _start_targets(seq_off, item, event_type, event_source):
    """Targets of the start token: the first item, and the first item opened from a feed (-1 if none)."""
    n = len(seq_off) - 1
    fe = np.full(n, -1, np.int32)
    new = np.full(n, -1, np.int32)
    for s in range(n):
        a, b = seq_off[s], seq_off[s + 1]
        if b > a:
            new[s] = item[a]
        for j in range(a, b):
            if event_type[j] == 1 and event_source[j] == 1:
                fe[s] = item[j]
                break
    return fe, new


class CorpusOnGpu:
    """Token and target arrays of a corpus on the device, and the batch sampler."""

    def __init__(self, device: torch.device, corpus: str, n_feed_open_targets: int):
        folder = corpus_dir(corpus)
        self.exclude_test = corpus == "no_test"
        self.sequence_split = np.load(folder / "seq_split.npy")
        self.off_np = np.load(folder / "seq_off.npy")

        def on_device(name: str) -> torch.Tensor:
            return torch.from_numpy(np.load(folder / f"{name}.npy")).to(device)

        self.item, self.et, self.es = on_device("tok_item"), on_device("tok_et"), on_device("tok_es")
        self.tgt = {"fe": on_device("tgt_fe"), "new": on_device("tgt_new")}
        self.feed_opens = on_device("tgt_fe_multi")[:, :n_feed_open_targets]
        tokens = [np.load(folder / f"{name}.npy") for name in ("tok_item", "tok_et", "tok_es")]
        start_fe, start_new = _start_targets(self.off_np, *tokens)
        self.start_tgt = {"fe": torch.from_numpy(start_fe).to(device), "new": torch.from_numpy(start_new).to(device)}
        self.off = torch.from_numpy(self.off_np).to(device)
        self.lens_np = np.diff(self.off_np)
        self.lens = torch.from_numpy(self.lens_np).to(device)

    def batches(self, token_budget: int, rng: np.random.Generator, eval_oversampling: int) -> list[np.ndarray]:
        """One epoch of batches of sequence indices, in random order.

        ``eval_oversampling`` repeats the val and test sequences of the ``all``
        corpus (they are 2% of it, and they are what the encoder is used on).
        """
        lengths = self.lens_np
        split = self.sequence_split
        base = np.arange(len(lengths))
        if self.exclude_test:
            base = base[split != SPLITS.index("test")]
        if eval_oversampling > 1:
            labelled = np.where((split > 0) & ~((split == SPLITS.index("test")) & self.exclude_test))[0]
            base = np.concatenate([base] + [labelled] * (eval_oversampling - 1))
        perm = base[rng.permutation(len(base))]
        order = perm[np.argsort((lengths[perm] + LENGTH_BUCKET - 1) // LENGTH_BUCKET, kind="stable")]
        bucket = (lengths[order] + LENGTH_BUCKET - 1) // LENGTH_BUCKET
        out = []
        for b in np.unique(bucket):
            ids = order[bucket == b]
            size = max(1, token_budget // (int(b) * LENGTH_BUCKET + 1))
            out += [ids[i:i + size] for i in range(0, len(ids), size)]
        return [out[k] for k in rng.permutation(len(out))]


def padded_index(off: torch.Tensor, lens: torch.Tensor,
                 sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Token indices ``[B, L]`` of a batch of sequences, right-padded, with the padding mask and ``L``."""
    longest = int(lens[sequences].max().item())
    ar = torch.arange(longest, device=off.device)
    mask = ar[None, :] < lens[sequences][:, None]
    idx = torch.where(mask, off[sequences][:, None] + ar[None, :], 0)
    return idx, mask, longest


def sampled_softmax(model: TrajectoryEncoder, items: ItemTables, query: torch.Tensor, target: torch.Tensor,
                    pool: torch.Tensor, scale: torch.Tensor, id_drop: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss of ``query [N, d]`` against ``target [N]`` with the negatives ``pool [K]``.

    Also returns the share of negatives that the positive beats.
    """
    vp = model.out_vec(items.rep(model, target, id_drop=id_drop))
    vn = model.out_vec(items.rep(model, pool, id_drop=id_drop))
    lp = scale * (query * vp).sum(-1) - items.logq[target.long()] + model.bias(items.slot[target.long()])
    ln = scale * (query @ vn.T) - items.logq[pool.long()][None, :] + model.bias(items.slot[pool.long()])[None, :]
    ln = ln.masked_fill(pool[None, :] == target[:, None], -1e4)
    loss = torch.logsumexp(torch.cat([lp[:, None], ln], dim=1), dim=1) - lp
    with torch.no_grad():
        beaten = (lp[:, None] > ln).float().mean(1)
    return loss.mean(), beaten.mean()


def build_model(n_id: int, model_cfg: dict, variant_cfg: dict) -> TrajectoryEncoder:
    """The encoder for the recipe's ``[model]`` table and one of its ``[variants]``."""
    return TrajectoryEncoder(n_id, d=model_cfg["d"], nl=model_cfg["layers"], nh=model_cfg["heads"],
                             d_id=variant_cfg["d_id"], drop=model_cfg["dropout"], heads=HEADS,
                             id_bias=variant_cfg["id_bias"])


def _head_targets(head: str, data: CorpusOnGpu, idx: torch.Tensor, mask: torch.Tensor, sequences: torch.Tensor,
                  states: torch.Tensor, longest: int, max_positions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(states, positives)`` of one head: the positions of a batch that have a target, at most ``max_positions``.

    Position 0 is the start token. The ``fe`` target is one of the next few
    new items opened from a feed, drawn uniformly at random.
    """
    if head == "fe":
        opens = data.feed_opens[idx]
        n_valid = (opens >= 0).sum(-1)
        pick = (torch.rand(n_valid.shape, device=idx.device) * n_valid).long().clamp_max(opens.shape[-1] - 1)
        targets = torch.where(mask & (n_valid > 0), opens.gather(-1, pick[..., None])[..., 0], -1)
    else:
        targets = torch.where(mask, data.tgt[head][idx], -1)
    targets = torch.cat([data.start_tgt[head][sequences][:, None], targets], dim=1)
    valid = targets >= 0
    chosen, positives = states[:, :longest + 1][valid], targets[valid]
    if len(positives) > max_positions:
        keep = torch.randperm(len(positives), device=idx.device)[:max_positions]
        chosen, positives = chosen[keep], positives[keep]
    return chosen, positives


def _negative_pool(feed_open_positives: torch.Tensor, new_targets: torch.Tensor, items: ItemTables,
                   cfg: dict) -> torch.Tensor:
    """The negatives shared by both heads in one step.

    Draws from the batch's ``fe`` positives, from its ``new`` targets and
    uniformly from the corpus items, in that order. A source that is empty in
    this batch is replaced by the corpus items, so the pool keeps its size.
    """
    corpus_items = items.corpus_items
    device = corpus_items.device
    parts = []
    for source, n in ((feed_open_positives, cfg["negatives_feed_open"]), (new_targets, cfg["negatives_new_item"]),
                      (corpus_items, cfg["negatives_uniform"])):
        source = source if len(source) else corpus_items
        parts.append(source[torch.randint(len(source), (n,), device=device)])
    return torch.cat(parts)


def fit_encoder(corpus: str, variant: str, recipe: dict, device: torch.device,
                log: Callable[[str], None] = print) -> TrajectoryEncoder:
    """Train one model and return it. ``recipe`` is the parsed ``configs/recipes/encoder.toml``.

    Float32 products use TF32 during the run, as in the runs that trained the
    stored checkpoints.
    """
    with tf32_matmul(True, cudnn=True):
        return _fit(corpus, variant, recipe, device, log)


def _fit(corpus: str, variant: str, recipe: dict, device: torch.device,
         log: Callable[[str], None]) -> TrajectoryEncoder:
    """The body of :func:`fit_encoder`: set-up, then the step loop."""
    cfg, model_cfg = recipe["training"], recipe["model"]
    oversampling = recipe["corpora"][corpus]["eval_oversampling"]
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    started = time.time()

    items = ItemTables(device, corpus, model_cfg["min_item_count"], content=recipe["corpora"][corpus]["content_table"])
    items.set_sampling_correction(cfg["negatives_feed_open"], cfg["negatives_new_item"], cfg["negatives_uniform"])
    data = CorpusOnGpu(device, corpus, cfg["feed_open_targets"])
    model = build_model(items.n_id, model_cfg, recipe["variants"][variant]).to(device)
    id_drop = cfg["id_dropout"]
    dense = [p for name, p in model.named_parameters() if not name.startswith(("id_emb", "id_bias"))]
    sparse = list(model.id_emb.parameters()) + (list(model.id_bias.parameters()) if model.id_bias is not None else [])
    opt = torch.optim.AdamW(dense, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    opt_id = torch.optim.SparseAdam(sparse, lr=cfg["id_lr"])
    log(f"{corpus}_{variant}: {sum(p.numel() for p in dense) / 1e6:.2f}M dense parameters, "
        f"{items.n_id:,} ID rows, {len(data.lens_np):,} sequences, setup {time.time() - started:.0f} s")

    epoch_batches = data.batches(cfg["token_budget"], rng, oversampling)
    total = int(len(epoch_batches) * cfg["epochs"])
    warmup = cfg["warmup_steps"]

    def lr_factor(step: int) -> float:
        """Warm-up times a cosine decay to 10% of the peak."""
        return min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))

    def head_loss(head: str, states: torch.Tensor, positives: torch.Tensor, pool: torch.Tensor,
                  scale: torch.Tensor, log_now: bool) -> torch.Tensor:
        """Sampled-softmax loss of one head (0 when the batch has no target for it)."""
        if len(positives) == 0:
            return torch.zeros((), device=device)
        query = model.query(states.float(), head)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, beaten = sampled_softmax(model, items, query, positives, pool, scale, id_drop)
        if log_now:
            recent.setdefault(f"loss_{head}", []).append(loss.item())
            recent.setdefault(f"beaten_{head}", []).append(beaten.item())
        return loss

    log(f"{len(epoch_batches):,} batches per epoch, {total:,} steps")
    new_item_weight = cfg["new_item_weight"]
    step, epoch = 0, 0
    recent: dict[str, list[float]] = {}
    last_log = time.time()
    while step < total:
        if epoch > 0:
            epoch_batches = data.batches(cfg["token_budget"], rng, oversampling)
        for batch in epoch_batches:
            if step >= total:
                break
            for group in opt.param_groups:
                group["lr"] = cfg["lr"] * lr_factor(step)
            for group in opt_id.param_groups:
                group["lr"] = cfg["id_lr"] * lr_factor(step)
            sequences = torch.from_numpy(batch).to(device)
            idx, mask, longest = padded_index(data.off, data.lens, sequences)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rep = items.rep(model, torch.where(mask, data.item[idx], 0), valid=mask, id_drop=id_drop)
                h = model.encode(rep, data.et[idx].long() * mask, data.es[idx].long() * mask)
            scale = model.logit_scale.exp().clamp(max=100.0)
            log_now = step % 50 == 0
            # The order of the random draws (fe targets, the pool, the fe loss, then the new head) is the
            # order of the runs that trained the stored checkpoints.
            fe_states, fe_positives = _head_targets("fe", data, idx, mask, sequences, h, longest,
                                                    cfg["max_positions"])
            new_targets = torch.where(mask, data.tgt["new"][idx], -1)
            pool = _negative_pool(fe_positives, new_targets[new_targets >= 0], items, cfg)
            loss = 1.0 * head_loss("fe", fe_states, fe_positives, pool, scale, log_now)
            new_states, new_positives = _head_targets("new", data, idx, mask, sequences, h, longest,
                                                      cfg["max_positions"])
            loss = loss + new_item_weight * head_loss("new", new_states, new_positives, pool, scale, log_now)
            opt.zero_grad(set_to_none=True)
            opt_id.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense, 1.0)
            opt.step()
            opt_id.step()
            step += 1
            if step % LOG_EVERY == 0:
                means = " ".join(f"{k} {float(np.mean(v)):.4f}" for k, v in recent.items())
                rate = LOG_EVERY * cfg["token_budget"] / (time.time() - last_log)
                log(f"step {step}/{total} epoch {epoch} {means} scale {scale.item():.1f} "
                    f"lr {cfg['lr'] * lr_factor(step):.2e} {rate / 1e6:.2f}M tokens/s {time.time() - started:.0f} s")
                recent.clear()
                last_log = time.time()
        epoch += 1
    log(f"trained {step} steps in {time.time() - started:.0f} s")
    return model


def train_encoder(corpus: str, variant: str, recipe: dict, device: torch.device,
                  log: Callable[[str], None] = print) -> Path:
    """Train one model and write its checkpoint (``{"model": state_dict}``); refuses to replace a stored one."""
    target = checkpoint_file(f"{corpus}_{variant}")
    if target.exists():
        raise FileExistsError(f"{target} exists; training would replace a stored checkpoint")
    model = fit_encoder(corpus, variant, recipe, device, log)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, target)
    log(f"saved {target}")
    return target
