"""Freeze-day check of the files about to be selected as final submissions.

    python scripts/freeze_check.py configs/ensembles/fd16p20_m04_02.toml configs/ensembles/w8p20.toml

For every recipe: print what the records say about its file, compare the
sha256 of the stored submission with the sha256 on record, re-run the gate on
the stored files and compare its text with the gate output stored when the
file was built (``avito_ranker/validation/freeze.py``).

The gate reads the member logs from ``--logs-dir``, by default the registered
``outputs.member_logs``, which holds the logs of the final members. Steps 4
and 7 write their logs to ``<outputs.runs>/logs``; a stored file is checked
against the stored logs. The first lines name the folders read.

Today's gate texts are written to ``--out-dir`` (default: a folder named by
the UTC time under ``<outputs.runs>/freeze_check/``). The exit code is 0 only
if every sha256 equals its record and every gate passes. CPU only,
``--threads`` threads (default 12), about 4 s per file. Nothing is submitted.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:   # the package is imported in main, after the thread caps are set
    from avito_ranker.validation.freeze import FreezeResult


def describe(result: FreezeResult, recorded_sha256: str, recorded_gate: Path) -> list[str]:
    """The lines printed for one candidate."""
    if result.sha256_matches:
        lines = [f"{result.name}: sha256 equals the record ({result.sha256})"]
    else:
        lines = [f"{result.name}: SHA256 MISMATCH: file {result.sha256}, record {recorded_sha256}"]
    lines.append(f"{result.name}: gate {result.gate_verdict} -> {result.gate_output}")
    if result.gate_matches_record:
        lines.append(f"{result.name}: gate text identical to the record ({recorded_gate})")
    else:
        lines.append(f"{result.name}: gate text DIFFERS from the record ({recorded_gate}):")
        lines.extend(f"    {line}" for line in result.gate_diff)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipes", type=Path, nargs="+", help="ensemble recipes, files of configs/ensembles/")
    parser.add_argument("--logs-dir", metavar="DIR", type=Path, default=None,
                        help="folder of the member logs (default: outputs.member_logs of configs/paths.toml; "
                             "steps 4 and 7 write to <outputs.runs>/logs)")
    parser.add_argument("--out-dir", metavar="DIR", type=Path, default=None,
                        help="folder for today's gate texts (default: <outputs.runs>/freeze_check/<UTC time>)")
    parser.add_argument("--threads", metavar="N", type=int, default=12,
                        help="CPU threads of polars and numba (default: 12)")
    args = parser.parse_args()

    from avito_ranker.threads import limit_threads

    limit_threads(args.threads)
    from avito_ranker.ensemble.config import load_ensemble_config
    from avito_ranker.paths import path, shown
    from avito_ranker.validation.freeze import candidate_table, check_candidate, recorded_gate_output

    configs = [load_ensemble_config(p) for p in args.recipes]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or path("outputs.runs") / "freeze_check" / stamp
    logs_dir = args.logs_dir or path("outputs.member_logs")

    print(f"submissions from {shown(path('outputs.submissions'))}, pools from {shown(path('outputs.predictions'))}")
    print(f"member logs from {shown(logs_dir)}")
    print("candidates on record:")
    for line in candidate_table(configs):
        print(f"  {line}")
    all_ok = True
    for config in configs:
        result = check_candidate(config, out_dir, logs_dir)
        for line in describe(result, config.record.sha256, recorded_gate_output(config)):
            print(line, flush=True)
        all_ok = all_ok and result.ok
    print(f"freeze check {'passed' if all_ok else 'FAILED'} at {stamp}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
