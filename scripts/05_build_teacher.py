"""Step 5: the distillation teacher of the students.

    python scripts/05_build_teacher.py dump <member>.pt [<member>.pt ...] --out-file teachers/teacher_x.npy
    python scripts/05_build_teacher.py average configs/recipes/teacher_mean11.toml [--out-file FILE]

``dump`` scores every train candidate with each hard-label member and writes
the sum of their scores (``teacher_x.npy``, 987 MB) with the member count
(``teacher_x.n``). About 105 s per member on one GPU, plus about a
minute to load the train split for each input view; 50 GB of GPU memory.

``average`` reads the group dumps a teacher recipe lists (under
``teachers.dumps`` of ``configs/paths.toml``) and writes the teacher: the
summed scores divided by the member count, centred inside each feed. It runs
on the CPU in about 10 s with the dumps in the page cache (10 GB to read for the
25-member teacher) and needs about 8 GB of memory. The default output
is the recipe's ``[teachers]`` entry; existing files are never replaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from avito_ranker.training.teacher import TeacherRecipe, build_teacher, dump_train_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    dump = commands.add_parser("dump", help="sum the train scores of hard-label members")
    dump.add_argument("checkpoints", type=Path, nargs="+", help="checkpoints of the hard-label members")
    dump.add_argument("--out-file", metavar="FILE", type=Path, required=True,
                      help="output .npy; the member count goes to the same name with suffix .n")
    dump.add_argument("--device", default="cuda:0", help="GPU to score on (default: cuda:0)")
    average = commands.add_parser("average", help="build a teacher file from group dumps")
    average.add_argument("recipe", type=Path, help="teacher recipe, configs/recipes/teacher_mean*.toml")
    average.add_argument("--out-file", metavar="FILE", type=Path, default=None,
                         help="output .npy (default: the recipe's [teachers] entry of configs/paths.toml)")
    args = parser.parse_args()

    if args.command == "dump":
        dump_train_scores(args.checkpoints, args.out_file, args.device)
        print(f"wrote {args.out_file} ({len(args.checkpoints)} members)")
    else:
        recipe = TeacherRecipe.read(args.recipe)
        target = build_teacher(recipe, args.out_file)
        print(f"wrote {target}: teacher {recipe.name}, {recipe.n_members} members in {len(recipe.dumps)} dumps")


if __name__ == "__main__":
    main()
