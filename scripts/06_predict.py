"""Step 6: score val and test with a trained member.

    python scripts/06_predict.py --checkpoint <member>.pt --member <member>
    python scripts/06_predict.py --checkpoint <member>.pt --member <member> --splits val --out-dir rebuild/

Writes ``predictions/nn_<member>_trainonly_<split>.npy`` under ``--out-dir``
(default: ``outputs.runs`` of ``configs/paths.toml``), one float32 score per
candidate row of the split, the file the ensemble step reads for its
train-only branch. Existing files are never replaced. Step 8 reads members
from ``outputs.predictions`` unless given ``--members-dir
<out-dir>/predictions``. The checkpoint carries the list of pre-computed
inputs, so the script needs no recipe. It reads checkpoints written by this
package and the stored checkpoints of the competition members.

One GPU, about 11 s per member for both splits (val 1.39M rows, test 2.90M
rows; under 2 GB of GPU memory). The validation NDCG is printed. On the GPU
model that trained the members the output equals the stored predictions bit
for bit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from avito_ranker.data.loader import load_split
from avito_ranker.io import refuse_existing
from avito_ranker.metrics import ndcg
from avito_ranker.model import load_ranker
from avito_ranker.paths import path
from avito_ranker.training.predict import predict, save_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", metavar="FILE", type=Path, required=True, help="checkpoint of the member")
    parser.add_argument("--member", metavar="NAME", required=True, help="member name used in the output file names")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"],
                        help="splits to score (default: val test)")
    parser.add_argument("--out-dir", metavar="DIR", type=Path, default=None,
                        help="run folder: writes <DIR>/predictions (default: outputs.runs of configs/paths.toml)")
    parser.add_argument("--device", default="cuda:0", help="GPU to score on (default: cuda:0)")
    args = parser.parse_args()

    predictions = (args.out_dir or path("outputs.runs")) / "predictions"
    targets = {split: predictions / f"nn_{args.member}_trainonly_{split}.npy" for split in args.splits}
    try:
        refuse_existing(*targets.values())
    except FileExistsError as error:
        parser.error(str(error))
    predictions.mkdir(parents=True, exist_ok=True)

    model = load_ranker(args.checkpoint, args.device)
    n_parameters = sum(p.numel() for p in model.parameters())
    config = model.config
    print(f"{args.checkpoint.name}: {n_parameters:,} parameters, width {config.d_model}, "
          f"{config.view.n_inputs} pre-computed inputs, dense block of {config.n_dense} columns", flush=True)
    for split_name, target in targets.items():
        started = time.time()
        split = load_split(split_name, config.view, args.device)
        scores = predict(model, split)
        save_scores(target, scores)
        print(f"{split.summary()} -> {target} ({time.time() - started:.1f} s)", flush=True)
        if split.targets is not None:
            score = ndcg(scores, split.targets, split.candidate_offsets, split.feed_sizes)
            print(f"{split_name} NDCG {score:.5f}", flush=True)
        del split
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
