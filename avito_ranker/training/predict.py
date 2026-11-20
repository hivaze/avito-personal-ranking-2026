"""Scoring a split with a trained ranker.

The numerics follow the runs that wrote the member predictions, because the
ensemble is built from those files and a re-run should reproduce them:

* the model runs in bfloat16 mixed precision with its heads in float32;
* the dense block is computed inside the mixed-precision region (training
  computes it in float32 beforehand), so the candidate-to-history cosines that
  feed it are bfloat16 here;
* float32 matrix products use TF32, which the trainer and the fine-tune switch
  on for their whole run;
* batches hold up to 1,024 feeds and 65,536 padded candidates, twice the
  training size. The padded history length depends on the feeds that share a
  batch, and bfloat16 sums depend on it.

With these settings the stored val and test predictions of all 36 stored
member checkpoints come back bit for bit on the GPU model that trained them.
Turning TF32 off, or using the training batch size, moves single scores by up
to 0.03 and the validation NDCG of a member by up to 0.00004.

``save_scores`` writes a score file in the one format every later step reads
(float32, one value per candidate row) and never replaces an existing file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from avito_ranker.data.loader import SplitTensors
from avito_ranker.model.blocks import tf32_matmul
from avito_ranker.model.ranker import Ranker

__all__ = ["predict", "save_scores", "tf32_matmul"]


@torch.no_grad()
def predict(model: Ranker, split: SplitTensors, feed_ids: np.ndarray | None = None, *,
            max_candidates: int = 65536, max_feeds: int = 1024, tf32: bool = True) -> np.ndarray:
    """Return the model's scores as a float32 array aligned with the split's candidate rows.

    Rows of feeds outside ``feed_ids`` (default: all feeds) stay 0. The model
    is put in evaluation mode for the call and returned to its previous mode.

    Batches without a domain mix are scored with the model's inference domain,
    which is the case for val and test. A split that carries a domain mix (the
    train split when teacher scores are dumped) is scored with it.

    ``tf32=False`` computes float32 products at full precision. The teacher
    scores were dumped that way; every member prediction used TF32.
    """
    was_training = model.training
    model.eval()
    feed_ids = np.arange(split.n_feeds) if feed_ids is None else feed_ids
    scores = torch.zeros(split.n_candidates, dtype=torch.float32, device=split.device)
    with tf32_matmul(tf32):
        for feeds in split.plan_batches(feed_ids, max_candidates, max_feeds):
            batch = split.gather(feeds)
            with torch.autocast(split.device.type, dtype=torch.bfloat16):
                batch_scores = model(batch)
            split.scatter(scores, batch, batch_scores)
    model.train(was_training)
    return scores.cpu().numpy()


def save_scores(file: str | Path, scores: np.ndarray) -> Path:
    """Write ``scores`` as a float32 ``.npy`` file, creating its folder; refuses to replace an existing file."""
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("xb") as handle:
        np.save(handle, np.asarray(scores).astype(np.float32))
    return file
