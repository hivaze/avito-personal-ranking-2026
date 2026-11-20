"""Step 7: fine-tune a trained member on the validation days, out of fold.

    python scripts/07_finetune.py --checkpoint <member>.pt --member <member>
    python scripts/07_finetune.py --checkpoint <member>.pt --member <member> --out-dir rebuild/

Five folds of the validation feeds grouped by user, three seeds per fold, the
values of configs/recipes/finetune.toml (``--recipe``;
avito_ranker/training/finetune.py explains the method and its leak
discipline). With ``<tag>`` the recipe's ``file_tag`` it writes, under
``--out-dir`` (default: ``outputs.runs`` of ``configs/paths.toml``)

    predictions/nn_<member>_<tag>_val_oof.npy   out-of-fold score of every val candidate row (1,389,971 float32)
    predictions/nn_<member>_<tag>_test.npy      mean of the 15 fold models on every test row (2,904,806 float32)
    logs/<member>_<tag>_ft.log                  the log; the pre-submission gate reads its "extra features" line

These are the names the ensemble step reads; step 8 reads members from
``outputs.predictions`` unless given ``--members-dir <out-dir>/predictions``,
and step 9 reads logs from ``outputs.member_logs`` unless given ``--logs-dir
<out-dir>/logs``. The two prediction files are never replaced. The log is
appended to, so a fine-tune that crashed can be run again under the same
name; the gate reads the last input list in it. The checkpoint carries the
member's list of pre-computed inputs, so one command serves full-view and
popularity-light members alike. It reads checkpoints written by this package
and the stored member checkpoints.

One GPU: about 4 minutes per member (15 fold models), 5.1 GiB of GPU memory
and 4.3 GB of host memory, measured on an otherwise idle GPU. The log
prints the held-out NDCG of each fold before and after the fine-tune, and at
the end the out-of-fold val NDCG.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from avito_ranker.features.names import FINETUNE_EXTRAS
from avito_ranker.io import refuse_existing
from avito_ranker.model import read_checkpoint
from avito_ranker.paths import path
from avito_ranker.training.finetune import (
    default_recipe_file,
    finetune,
    load_finetune_recipe,
    load_finetune_splits,
)
from avito_ranker.training.predict import save_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", metavar="FILE", type=Path, required=True,
                        help="checkpoint of the member as it left training")
    parser.add_argument("--member", metavar="NAME", required=True, help="member name used in the output file names")
    parser.add_argument("--recipe", metavar="FILE", type=Path, default=None,
                        help="fine-tune recipe (default: configs/recipes/finetune.toml)")
    parser.add_argument("--out-dir", metavar="DIR", type=Path, default=None,
                        help="run folder: writes <DIR>/predictions and <DIR>/logs "
                             "(default: outputs.runs of configs/paths.toml)")
    parser.add_argument("--device", default="cuda:0", help="GPU to train on (default: cuda:0)")
    args = parser.parse_args()

    recipe_file = args.recipe or default_recipe_file()
    recipe = load_finetune_recipe(recipe_file)
    run_dir = args.out_dir or path("outputs.runs")
    stem = f"nn_{args.member}_{recipe.file_tag}"
    targets = {"val": run_dir / "predictions" / f"{stem}_val_oof.npy",
               "test": run_dir / "predictions" / f"{stem}_test.npy"}
    log_file = run_dir / "logs" / f"{args.member}_{recipe.file_tag}_ft.log"
    try:
        refuse_existing(*targets.values())
    except FileExistsError as error:
        parser.error(str(error))
    for folder in (targets["val"].parent, log_file.parent):
        folder.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a") as log_handle:
        def log(line: str) -> None:
            """Print a line and append it to the fine-tune log."""
            print(line, flush=True)
            log_handle.write(line + "\n")
            log_handle.flush()

        started = time.time()
        log(f"==== {time.ctime()} fine-tune of {args.member} from {args.checkpoint}")
        log(f"recipe {recipe_file}: {asdict(recipe)}")
        checkpoint = read_checkpoint(args.checkpoint)
        val, test = load_finetune_splits(checkpoint.config.view, args.device)
        log(val.summary())
        log(test.summary())
        log(f"extra features ({len(FINETUNE_EXTRAS)}): {list(FINETUNE_EXTRAS)}")  # the line the gate reads

        result = finetune(checkpoint, val, test, recipe, log)
        for split, scores in (("val", result.val_scores), ("test", result.test_scores)):
            save_scores(targets[split], scores)
            log(f"{split} -> {targets[split]}")
        memory = torch.cuda.max_memory_allocated(args.device) / 2**30 if torch.device(args.device).type == "cuda" else 0
        log(f"done in {(time.time() - started) / 60:.1f} min, peak GPU memory {memory:.1f} GiB")


if __name__ == "__main__":
    main()
