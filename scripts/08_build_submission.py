"""Step 8: pool the members of an ensemble, read it out on validation, write the submission.

    python scripts/08_build_submission.py configs/ensembles/fd16p20.toml
    python scripts/08_build_submission.py configs/ensembles/fd16p20.toml --write
    python scripts/08_build_submission.py configs/ensembles/fd16p20.toml --write --out-dir rebuild/
    python scripts/08_build_submission.py configs/ensembles/fd16p20.toml --members-dir artifacts/runs/predictions
    python scripts/08_build_submission.py configs/ensembles/fd16p20_m04_02.toml --write --out-dir rebuild/

The member predictions are read from ``--members-dir``, by default the
registered ``outputs.predictions``, which holds the stored predictions of the
final members. Steps 4, 6 and 7 write to ``<outputs.runs>/predictions``, so
pass ``--members-dir artifacts/runs/predictions`` for a rebuild from new runs.
The first lines name the folders read.

Without ``--write`` nothing is stored. The read-out is the validation NDCG of
the final score and of its two branches, then the paired difference to the
recipe's reference submission on plain validation and under the public and
private mix (``avito_ranker/validation/private_mix.py``).

``--write`` stores the four pool files and the submission parquet in the
registered predictions and submissions folders, or in ``--out-dir``. Existing
files are never replaced, and nothing is written unless all five can be. To
check a rebuild against a stored submission, write it to another folder and
compare the sha256.

A recipe with the feed-size rule (``fd16p20_m04_02.toml``) weights the
popularity-light pool by the number of candidates of each feed; the first
read-out line states the rule.

CPU only, ``--threads`` threads (default 12). 11 s for each of FD16P20,
FD16P20vf and W8P20 (20 to 24 members) with ``--write`` and the files in the
page cache, measured on 2026-11-16, and 11 s for FD16P20@m0.4_0.2 on
2026-11-18; about 7 s of it go to the cold-share table of the private mix.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:   # the package is imported in main, after the thread caps are set
    from avito_ranker.ensemble.build import Branches
    from avito_ranker.ensemble.config import EnsembleConfig


def read_out(config: EnsembleConfig, branches: dict[str, Branches]) -> list[str]:
    """Validation NDCG of the ensemble and its paired difference to the reference submission."""
    import numpy as np

    from avito_ranker.data.flatten import read_flat
    from avito_ranker.ensemble.build import final_score, load_pool
    from avito_ranker.ensemble.feeds import load_feed_index
    from avito_ranker.metrics import ndcg_per_feed
    from avito_ranker.validation import private_mix

    val, test = load_feed_index("val"), load_feed_index("test")
    labels = read_flat("val", "cand_target", mmap=False)
    weight = config.finetuned_weight

    def val_ndcg(scores: np.ndarray) -> np.ndarray:
        """Per-feed validation NDCG of a score column, cast to float32 first."""
        return ndcg_per_feed(np.asarray(scores, np.float32), labels, val.offsets, val.sizes)

    final = val_ndcg(final_score(branches["val"], val, weight))
    lines = [
        f"{config.name}: {len(config.full_view)} full-view + {len(config.popularity_light)} popularity-light members, "
        f"popularity-light weight {config.popularity_light_weight_text()}",
        f"val NDCG: final {final.mean():.5f} | fine-tuned branch {val_ndcg(branches['val'].finetuned).mean():.5f} "
        f"| train-only branch {val_ndcg(branches['val'].train_only).mean():.5f}",
    ]

    reference = val_ndcg(final_score(load_pool(config.reference_pool_tag, "val"), val, weight))
    mix = private_mix.compare(final, reference)
    test_final = final_score(branches["test"], test, weight)
    test_reference = final_score(load_pool(config.reference_pool_tag, "test"), test, weight)
    lines.append(
        f"against {config.reference_pool_tag} (val {reference.mean():.5f}): "
        f"val {mix.val.mean:+.5f} +- {mix.val.se:.5f} | public mix {mix.public.mean:+.5f} +- {mix.public.se:.5f} "
        f"| private mix {mix.private.mean:+.5f} +- {mix.private.se:.5f} "
        f"| test correlation {np.corrcoef(test_final, test_reference)[0, 1]:.4f}"
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path, help="ensemble recipe, a file of configs/ensembles/")
    parser.add_argument("--members-dir", metavar="DIR", type=Path, default=None,
                        help="folder of the member predictions (default: outputs.predictions of configs/paths.toml; "
                             "steps 4, 6 and 7 write to <outputs.runs>/predictions)")
    parser.add_argument("--write", action="store_true", help="store the pool files and the submission")
    parser.add_argument("--out-dir", metavar="DIR", type=Path, default=None,
                        help="with --write: folder for the five files (default: the registered folders)")
    parser.add_argument("--threads", metavar="N", type=int, default=12,
                        help="CPU threads of polars and numba (default: 12)")
    args = parser.parse_args()
    if args.out_dir is not None and not args.write:
        parser.error("--out-dir is only used together with --write")

    from avito_ranker.threads import limit_threads

    limit_threads(args.threads)
    from avito_ranker.ensemble.build import build_ensemble, write_ensemble
    from avito_ranker.ensemble.config import load_ensemble_config
    from avito_ranker.paths import path, shown
    from avito_ranker.validation.freeze import file_sha256

    members_dir = args.members_dir or path("outputs.predictions")
    print(f"member predictions from {shown(members_dir)}")
    print(f"reference pools from {shown(path('outputs.predictions'))}", flush=True)

    config = load_ensemble_config(args.recipe)
    branches = build_ensemble(config, members_dir)
    for line in read_out(config, branches):
        print(line, flush=True)

    if args.write:
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
        written = write_ensemble(config, branches, args.out_dir)
        for file in written:
            print(f"written {file}")
        print(f"sha256 {file_sha256(written[-1])}  {written[-1].name}")


if __name__ == "__main__":
    main()
