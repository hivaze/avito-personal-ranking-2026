"""Shared helpers: prediction, evaluation, feature-normalization stats, logging."""
import json
import sys
import time
from pathlib import Path

import os

import numpy as np
import torch

# Safety: refuse to run without an explicit single CUDA_VISIBLE_DEVICES.
# nn_x/w5 copy (wave 5, from nn_x/r6 2026-11-12 19:01 UTC): exactly one GPU id; lane.sh allows only the GPUs the wave-5
# task granted (<ids>), at most 4 compute processes per GPU and >= 60 GB free. CPU unit tests never import this module.
if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in tuple(str(i) for i in range(8)):  # some GPUs authorised 2026-11-12
    raise RuntimeError("nn_x/w5: set CUDA_VISIBLE_DEVICES to exactly one GPU id 0-7")

# CPU thread cap per process. r6 review fix: honour OMP_NUM_THREADS (queue_morning.sh exports 12) instead of forcing 16
torch.set_num_threads(min(16, int(os.environ.get("OMP_NUM_THREADS", "16") or 16)))

sys.path.insert(0, "<repo>/src")
from metrics import ndcg  # noqa: E402

from model import dense_features  # noqa: E402

ROOT = Path("<repo>")
# nn_x/w9/dw3 copy (wave 9 arm DW3, 2026-11-13): every output goes to nn_x/w9/dw3/{ckpt,logs,preds} (holdout preds:
# nn_x/w9/dw3/preds/holdout); the name W5 is kept so that the byte-identical w5 modules importing it keep working
W5 = ROOT / "nn_x" / "w9" / "dw3"
CKPT = W5 / "ckpt"
LOGS = W5 / "logs"
PREDS = W5 / "preds"
for p in (CKPT, LOGS, PREDS):
    p.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


@torch.no_grad()
def predict(model, data, feed_ids=None, cand_budget=65536, max_feeds=1024):
    """Return float32 numpy scores aligned with the split's flat candidate arrays
    (rows of feeds not in feed_ids stay 0)."""
    model.eval()
    feed_ids = np.arange(data.n) if feed_ids is None else feed_ids
    out = torch.zeros(data.n_cand, dtype=torch.float32, device=data.device)
    for fb in data.batches(feed_ids, cand_budget, max_feeds, shuffle=False):
        b = data.gather(fb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b, use_pos=False)
        data.scatter(out, b, lg)
    model.train()
    return out.cpu().numpy()


@torch.no_grad()
def eval_subset(model, data, feed_ids, rows=None):
    """Full NDCG of `model` on a subset of feeds of a (large) split without copying all scores to the host."""
    feed_ids = np.sort(feed_ids)
    M, off = data.M_np[feed_ids], data.cand_off_np[feed_ids]
    if rows is None:
        rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off, M)])
    out = torch.zeros(data.n_cand, dtype=torch.float32, device=data.device)
    model.eval()
    for fb in data.batches(feed_ids, 65536, 1024, shuffle=False):
        b = data.gather(fb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b, use_pos=False)
        data.scatter(out, b, lg)
    model.train()
    sc = out[torch.from_numpy(rows).to(data.device)].cpu().numpy()
    offs = np.concatenate([[0], np.cumsum(M)[:-1]])
    return ndcg(sc, data.target_np[rows], offs, M)


def eval_ndcg(scores, data, feed_ids=None):
    """Full NDCG over the given feeds (all feeds by default)."""
    M, off = data.M_np, data.cand_off_np
    if feed_ids is None:
        return ndcg(scores, data.target_np, off, M)
    feed_ids = np.sort(feed_ids)
    rows = np.concatenate([np.arange(off[f], off[f] + M[f]) for f in feed_ids])
    Ms = M[feed_ids]
    offs = np.concatenate([[0], np.cumsum(Ms)[:-1]])
    return ndcg(scores[rows], data.target_np[rows], offs, Ms)


@torch.no_grad()
def feature_stats(data, feed_ids, n_batches=40, seed=0, n_dense=None, dense_fn=None):
    """Per-feature mean/std of the dense features over valid candidates of a few random batches."""
    rng = np.random.default_rng(seed)
    bs = data.batches(feed_ids, shuffle=True, rng=rng)[:n_batches]
    s = s2 = n = 0
    for fb in bs:
        b = data.gather(fb)
        X = (dense_fn(b) if dense_fn is not None else dense_features(b, n_dense))[b["cmask"]].double()
        s = s + X.sum(0)
        s2 = s2 + (X ** 2).sum(0)
        n += X.shape[0]
    mean = s / n
    std = (s2 / n - mean ** 2).clamp_min(0).sqrt()
    return mean.float(), std.float()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))
