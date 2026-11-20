"""Step 2: build the pre-computed inputs that need no GPU.

    python scripts/02_build_features.py counts covisitation context exposure exploration
    python scripts/02_build_features.py encoder extras               # after scripts/03_train_encoder.py
    python scripts/02_build_features.py context --splits val test --out-dir rebuild/

Families (any subset, in any order):

  counts        it_hist_all_luo, it_hist_et1..4_luo            train, val, test     1 min
  covisitation  seq_cv_invd20, seq_cv_pop, seq_cv_norm         train, val, test    17 min
  context       ctx_nd_* (10), ctx_hn_* (6), ctx_hs_* (4)      train, val, test     4 min
  exposure      xp_imp_c50 (popularity-light view)             train, val, test     1 min
  exploration   exploration probability of every train feed    one .npz            11 min
  encoder       pre2_* (6), from stored raw encoder scores     train, val, test     1 min
  extras        the 42 fine-tune extras                        val, test            2 min

Times were measured on 2026-11-16 with 12 threads, building every family
against the stored arrays (36 min in all, 43 GB of host memory at the peak). Arrays go to ``<out-dir>/<split>/<name>.npy``;
the ranker reads them from the folder registered as ``data.feature_cache``,
which is the default of ``--out-dir``. An existing array is left alone when
the build gives the same values and is never replaced by other values unless
``--overwrite`` is given, so running the script against the stored arrays
checks them. A replaced array is unlinked first: the files of
``data.feature_cache`` are hard links into ``data.features``, and the copy
there keeps its stored values.

``exploration`` scores the train feeds with the stored position proxy
(``exploration.position_proxy``), which reproduces the stored probabilities
bit for bit, and writes ``exploration.probability`` (with ``--out-dir``:
``<out-dir>/exploration_probability.npz``). ``--refit-proxy`` fits a new
proxy first (2 more minutes). LightGBM training is not run-to-run
reproducible (docs/10-reproduce.md has the measurements). The fit runs 40
threads whatever ``--threads`` says, as the stored fit did; on a shared
machine, start the script under ``taskset``.

``--threads`` caps polars and numba and the LightGBM scoring. The script sets
``OPENBLAS_NUM_THREADS=1`` before numpy loads: the context kernels call BLAS
from inside their own threads.
"""
from __future__ import annotations

import argparse
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

FAMILIES = ("counts", "covisitation", "context", "exposure", "exploration", "encoder", "extras")
RANKER_EVENT_TYPES = (0, 1, 2, 3, 4)  # 0 = all events

Save = Callable[[str, dict], None]
Log = Callable[[str], None]


def build_counts(args: argparse.Namespace, out: Path, save: Save, log: Log) -> None:
    """Leave-user-out history counts (``avito_ranker.features.item_counts``)."""
    from avito_ranker.features import item_counts

    tables = item_counts.load_count_tables(RANKER_EVENT_TYPES)
    for split in args.splits:
        save(split, item_counts.leave_user_out_counts(split, tables, RANKER_EVENT_TYPES))


def build_covisitation(args: argparse.Namespace, out: Path, save: Save, log: Log) -> None:
    """Co-visitation scores; train rows are built day by day (``avito_ranker.features.covisitation``)."""
    from avito_ranker.features import covisitation

    eval_splits = [s for s in args.splits if s != "train"]
    if eval_splits:
        for split, columns in covisitation.build_eval(eval_splits).items():
            save(split, columns)
    if "train" in args.splits:
        save("train", covisitation.build_train(log=log))


def build_context(args: argparse.Namespace, out: Path, save: Save, log: Log) -> None:
    """In-feed context inputs (``avito_ranker.features.feed_context``)."""
    from avito_ranker.features import feed_context

    for split in args.splits:
        save(split, feed_context.build_feed_context(split))


def build_exposure(args: argparse.Namespace, out: Path, save: Save, log: Log) -> None:
    """The exposure input of the popularity-light view (``avito_ranker.features.exposure``)."""
    from avito_ranker.features import exposure
    from avito_ranker.features.names import EXPOSURE_INPUT

    for split, values in exposure.build_exposure(tuple(args.splits)).items():
        save(split, {EXPOSURE_INPUT: values})


def build_exploration(args: argparse.Namespace, out: Path, save: Save, log: Log) -> None:
    """The exploration probability of every train feed (``avito_ranker.features.exploration``)."""
    import lightgbm as lgb

    from avito_ranker.features import exploration
    from avito_ranker.io import clear_for_rewrite
    from avito_ranker.paths import path

    if args.refit_proxy:
        proxy_file = path("exploration.position_proxy") if args.out_dir is None else out / "position_proxy.txt"
        if proxy_file.exists() and not args.overwrite:
            raise FileExistsError(f"{proxy_file} exists; refitting would replace it (use --overwrite)")
        proxy = exploration.fit_position_proxy()
        clear_for_rewrite(proxy_file, args.overwrite)
        proxy.save_model(str(proxy_file))
        log(f"fitted the position proxy, saved {proxy_file}")
    else:
        proxy_file = path("exploration.position_proxy")
        if not proxy_file.exists():
            raise FileNotFoundError(f"{proxy_file} is missing; fit a new proxy with --refit-proxy")
        proxy = lgb.Booster(model_file=str(proxy_file))
        log(f"read the position proxy {proxy_file}")
    arrays = exploration.exploration_probability(proxy, args.threads, log)
    target = path("exploration.probability") if args.out_dir is None else out / "exploration_probability.npz"
    exploration.save_exploration_probability(target, arrays, args.overwrite)
    log(f"wrote {target}")


def build_encoder(args: argparse.Namespace, out: Path, save: Save, log: Log) -> None:
    """The ranker's encoder inputs from the stored raw scores (``trajectory_encoder.inputs``)."""
    from avito_ranker.features.trajectory_encoder import inputs as encoder_inputs
    from avito_ranker.features.trajectory_encoder.files import recipe_file

    recipe = tomllib.loads((args.recipe or recipe_file()).read_text())
    for split in args.splits:
        save(split, encoder_inputs.ranker_inputs(split, recipe["scoring"]["sample_feeds"]))


def build_extras(args: argparse.Namespace, out: Path, save: Save, log: Log) -> None:
    """The 42 fine-tune extras, val and test only (``avito_ranker.features.finetune_extras``)."""
    from avito_ranker.features import finetune_extras

    eval_splits = tuple(s for s in args.splits if s != "train")
    if eval_splits:
        for split, columns in finetune_extras.build_finetune_extras(eval_splits).items():
            save(split, columns)


BUILDERS = {
    "counts": build_counts,
    "covisitation": build_covisitation,
    "context": build_context,
    "exposure": build_exposure,
    "exploration": build_exploration,
    "encoder": build_encoder,
    "extras": build_extras,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("families", metavar="FAMILY", nargs="+", choices=FAMILIES,
                        help=f"input families to build, any of: {', '.join(FAMILIES)} (table above)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"],
                        help="splits to build (default: train val test)")
    parser.add_argument("--out-dir", metavar="DIR", type=Path, default=None,
                        help="output folder (default: data.feature_cache of configs/paths.toml)")
    parser.add_argument("--threads", metavar="N", type=int, default=12,
                        help="CPU threads of polars, numba and the LightGBM scoring (default: 12)")
    parser.add_argument("--overwrite", action="store_true", help="replace stored arrays that hold other values")
    parser.add_argument("--refit-proxy", action="store_true",
                        help="exploration: fit a new position proxy instead of reading the stored one")
    parser.add_argument("--recipe", metavar="FILE", type=Path, default=None,
                        help="encoder: recipe with the scoring settings (default: configs/recipes/encoder.toml)")
    args = parser.parse_args()

    from avito_ranker.threads import limit_threads

    limit_threads(args.threads, openblas_threads=1)   # the context kernels call BLAS from their own threads
    from avito_ranker.features.store import save_inputs, set_threads
    from avito_ranker.paths import path

    set_threads(args.threads)
    out = args.out_dir or path("data.feature_cache")
    started = time.time()

    def log(message: str) -> None:
        """Print a line stamped with the seconds since the start."""
        print(f"[{time.time() - started:6.0f} s] {message}", flush=True)

    def save(split: str, columns: dict) -> None:
        """Write the arrays of one split (see ``features.store.save_inputs``) and log their names."""
        names = save_inputs(out, split, columns, args.overwrite)
        log(f"{split}: {len(names)} arrays in place ({names[0]} ... {names[-1]})")

    for family in args.families:
        BUILDERS[family](args, out, save, log)
        log(f"{family} done")


if __name__ == "__main__":
    main()
