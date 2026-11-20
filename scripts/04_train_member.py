"""Step 4: train one ensemble member.

    python scripts/04_train_member.py configs/recipes/student_wide.toml --seed 951 --member student_wide_s951
    python scripts/04_train_member.py configs/recipes/teacher_hard_label.toml --seed 951 --member teacher_s951
    python scripts/04_train_member.py configs/recipes/student_wide.toml --seed 1 --member trial --epochs 0.05

A distilled student (student_*.toml) needs its teacher file (step 5). The
member name only labels the outputs; the stored runs of a recipe are listed
in its ``[record]`` table, and ``configs/ensembles/*.toml`` names them the
same way. ``--epochs`` shortens a trial run: the cooldown keeps its share of
the steps, the warmup its 500 steps. A holdout recipe chooses its checkpoint
from step 12,000 on (9,835 steps per epoch); a trial run with fewer steps
(``--epochs`` below 1.22) keeps its last step instead, and the script says
so.

Writes ``checkpoints/<member>.pt``, ``logs/<member>.log`` and
``predictions/nn_<member>_trainonly_{val,test}.npy`` under ``--out-dir``
(default: ``outputs.runs`` of ``configs/paths.toml``). It refuses to replace
the checkpoint or the predictions; a log that exists is appended to, and the
gate audits the last run in it. Step 8 reads members from
``outputs.predictions`` unless given ``--members-dir <out-dir>/predictions``;
step 9 reads logs from ``outputs.member_logs`` unless given ``--logs-dir
<out-dir>/logs``.

One GPU with at least 55 GiB of free memory: the train split stays on the GPU
(``avito_ranker/data/loader.py`` has the measured sizes). A student of the
final submissions trained for 221 to 259 minutes on a GPU; a run
with the holdout, 180 to 195 minutes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from avito_ranker.training.train import TrainingRecipe, run_length, train_member


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path,
                        help="training recipe: configs/recipes/student_*.toml or teacher_hard_label*.toml")
    parser.add_argument("--seed", type=int, required=True, help="seed of the run (the final members used 951 to 958)")
    parser.add_argument("--member", metavar="NAME", required=True,
                        help="name of the run; labels its checkpoint, log and predictions")
    parser.add_argument("--out-dir", metavar="DIR", type=Path, default=None,
                        help="run folder: writes <DIR>/checkpoints, <DIR>/logs and <DIR>/predictions "
                             "(default: outputs.runs of configs/paths.toml)")
    parser.add_argument("--epochs", type=float, default=None,
                        help="train for this many epochs instead of the recipe's (trial runs)")
    parser.add_argument("--device", default="cuda:0", help="GPU to train on (default: cuda:0)")
    args = parser.parse_args()

    recipe = TrainingRecipe.read(args.recipe)
    if args.epochs is not None:
        recipe = replace(recipe, epochs=args.epochs)
        if recipe.holdout_feeds:
            total = run_length(recipe)
            if recipe.select_from_step > total:
                print(f"trial run of {total} steps, select_from_step {recipe.select_from_step} lies past its end: "
                      f"the last step is kept", flush=True)
                recipe = replace(recipe, select_from_step=total)
    files = train_member(recipe, args.seed, args.member, args.out_dir, args.device)
    print(f"checkpoint {files.checkpoint}\nlog {files.log}\n" + "\n".join(map(str, files.predictions.values())))


if __name__ == "__main__":
    main()
