"""Step 3: the self-supervised trajectory encoder, from corpus to raw scores.

    python scripts/03_train_encoder.py corpus             # CPU: all corpora and both content tables
    python scripts/03_train_encoder.py train  all_id64    # GPU: one model; ten in total
    python scripts/03_train_encoder.py score  all_id64    # GPU: every scoring target of that model

Models are ``<corpus>_<variant>`` with corpus in all, no_test, window1,
window2, window3 and variant in id64, id128_bias. ``score`` writes the six raw
scores per target and content table; ``scripts/02_build_features.py encoder
extras`` then turns them into inputs.

What each model scores:

  no_test_*   val (full and partial content table)
  all_*       test and val (both tables; val only as the scale reference), and the 60,000-feed
              sample of each train block (scale reference)
  window<k>_* every train feed of block k

Settings come from ``configs/recipes/encoder.toml`` (``--recipe``). One GPU per
process, ``--device`` (default ``cuda:0``): scoring peaked at 63 GB of GPU
memory, and a short training run of an ``all`` model was seen at 102 GB. Training took 13 to 59
minutes per model (5.2 GPU-hours for the ten); scoring a model takes one to
three minutes, most of it loading the 7 to 21 GB checkpoint.

A stored file that already holds the values a step produces is left alone,
and one that holds other values is replaced only with ``--overwrite``
(``corpus`` and ``score``); ``train`` always refuses to replace a checkpoint.
The token arrays of the window corpora were deleted after the original
training; ``corpus`` writes them again (38 GiB), and ``score`` needs them for
the window models, because the rows of a model's ID table are numbered from
the item counts of its corpus.
"""
from __future__ import annotations

import argparse
import time
import tomllib
from pathlib import Path


def scoring_plan(model_name: str, scoring: dict) -> list[tuple[str, str, int]]:
    """``(content table, target, feeds per chunk)`` for every target a model scores."""
    from avito_ranker.features.trajectory_encoder.files import WINDOWS
    from avito_ranker.features.trajectory_encoder.score import split_model_name

    corpus, _ = split_model_name(model_name)
    if corpus.startswith("window"):
        return [("full", corpus, scoring["chunk_feeds_window"])]
    if corpus == "no_test":
        return [(content, "val", scoring["chunk_feeds_eval"]) for content in ("full", "partial")]
    chunk = scoring["chunk_feeds_eval"]
    plan = [(content, split, chunk) for content in ("full", "partial") for split in ("val", "test")]
    sample_chunks = scoring["chunk_feeds_sample"][model_name]
    return plan + [("full", f"window{k}_sample", sample_chunks[f"window{k}"]) for k in WINDOWS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("corpus", "train", "score"), help="what to do (see above)")
    parser.add_argument("model", nargs="?", help="<corpus>_<variant>, for train and score")
    parser.add_argument("--recipe", metavar="FILE", type=Path, default=None,
                        help="encoder recipe (default: configs/recipes/encoder.toml)")
    parser.add_argument("--device", default="cuda:0", help="train, score: GPU to use (default: cuda:0)")
    parser.add_argument("--threads", metavar="N", type=int, default=12,
                        help="CPU threads of polars, numba and torch (default: 12)")
    parser.add_argument("--overwrite", action="store_true",
                        help="corpus, score: replace stored files that hold other values")
    args = parser.parse_args()

    # Imported here so that --help answers without loading numba or torch, and after the thread caps.
    from avito_ranker.threads import limit_threads

    limit_threads(args.threads)
    from avito_ranker.features.store import set_threads
    from avito_ranker.features.trajectory_encoder.files import WINDOWS, recipe_file

    set_threads(args.threads)
    recipe = tomllib.loads((args.recipe or recipe_file()).read_text())
    started = time.time()

    def log(message: str) -> None:
        """Print a line stamped with the seconds since the start."""
        print(f"[{time.time() - started:6.0f} s] {message}", flush=True)

    if args.step == "corpus":
        from avito_ranker.features.trajectory_encoder import corpus

        log(f"all corpus: sequences per split {corpus.build_all_corpus(args.overwrite)}")
        for window in WINDOWS:
            log(f"window corpus: {corpus.build_window_corpus(window, args.overwrite)}")
        return

    if args.model is None:
        parser.error(f"{args.step} needs a model name")
    import torch

    from avito_ranker.features.trajectory_encoder import score
    from avito_ranker.features.trajectory_encoder.train import train_encoder

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if args.step == "train":
        corpus_name, variant = score.split_model_name(args.model)
        train_encoder(corpus_name, variant, recipe, device, log)
        return

    loaded: dict[str, tuple] = {}
    for content, target, chunk_feeds in scoring_plan(args.model, recipe["scoring"]):
        if content not in loaded:
            loaded.clear()
            torch.cuda.empty_cache()
            loaded[content] = score.load_encoder(args.model, content, recipe, device)
            log(f"loaded {args.model} with the {content} content table")
        model, items = loaded[content]
        split, feeds = score.target_feeds(target, recipe["scoring"]["sample_feeds"])
        scores = score.score_feeds(model, items, split, feeds, chunk_feeds)
        score.save_scores(content, target, args.model, scores, args.overwrite)
        log(f"scored {target}: {len(feeds):,} feeds, {len(scores['fe']):,} candidates")


if __name__ == "__main__":
    main()
