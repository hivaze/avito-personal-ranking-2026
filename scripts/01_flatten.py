"""Step 1: flatten the competition parquet into memory-mappable arrays.

    python scripts/01_flatten.py                      # train, val and test
    python scripts/01_flatten.py --splits val test --out-dir rebuild/

Reads ``<data.competition>/<split>/impressions/*.parquet`` and writes
``<data.flat>/<split>/`` (see ``avito_ranker/data/flatten.py`` for the layout),
or ``<out-dir>/<split>/``. CPU only, ``--threads`` threads (default 12): 17 s
for train and under a second each for val and test, measured on 2026-11-16
against the stored arrays.

An existing array is left alone when the rebuild gives the same values and is
replaced only with ``--overwrite``, so a run against the stored arrays checks
them. Every input, checkpoint row and prediction is aligned with these
arrays. A replaced file is unlinked first, so a hard-linked copy elsewhere
keeps its values.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS,
                        help="splits to flatten (default: train val test)")
    parser.add_argument("--out-dir", metavar="DIR", type=Path, default=None,
                        help="output folder (default: data.flat of configs/paths.toml)")
    parser.add_argument("--overwrite", action="store_true", help="replace stored arrays that hold other values")
    parser.add_argument("--threads", metavar="N", type=int, default=12,
                        help="CPU threads of polars (default: 12)")
    args = parser.parse_args()

    from avito_ranker.threads import limit_threads

    limit_threads(args.threads)
    from avito_ranker.data.flatten import flatten_split

    for split in args.splits:
        started = time.time()
        counts = flatten_split(split, out_dir=args.out_dir, overwrite=args.overwrite)
        print(f"{split}: {counts['feeds']:,} feeds, {counts['candidates']:,} candidates, "
              f"{counts['history_events']:,} history events, {time.time() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
