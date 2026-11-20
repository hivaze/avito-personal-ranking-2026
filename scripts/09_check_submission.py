"""Step 9: run the pre-submission gate on the submission of an ensemble recipe.

    python scripts/09_check_submission.py configs/ensembles/fd16p20.toml
    python scripts/09_check_submission.py configs/ensembles/fd16p20.toml --in-dir rebuild/ --logs-dir artifacts/runs/logs

The gate reads the submission parquet, the ensemble's pool files, the pool
files of the reference named in the recipe, and the logs of the members.
With ``--in-dir`` the submission and its pool files are taken from that folder
(a rebuild written by ``08_build_submission.py --write --out-dir``); the
reference pools are always the stored ones.

The member logs are read from ``--logs-dir``, by default the registered
``outputs.member_logs``, which holds the logs of the final members. Steps 4
and 7 write their logs to ``<outputs.runs>/logs``, so pass ``--logs-dir
artifacts/runs/logs`` to audit new runs. A member whose input list is not in
that folder fails the gate.

The folders read are printed first, on standard error, so that standard
output is the gate text alone and can be compared with a stored gate output.
The checks are described in ``avito_ranker/validation/gate.py``. The exit code
is 1 if any check fails, so a FAIL stops a shell pipeline. CPU only,
``--threads`` threads (default 12), about 4 s. Nothing is written and nothing
is submitted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path, help="ensemble recipe, a file of configs/ensembles/")
    parser.add_argument("--in-dir", metavar="DIR", type=Path, default=None,
                        help="folder with the submission and its pool files (default: the registered folders)")
    parser.add_argument("--logs-dir", metavar="DIR", type=Path, default=None,
                        help="folder of the member logs (default: outputs.member_logs of configs/paths.toml; "
                             "steps 4 and 7 write to <outputs.runs>/logs)")
    parser.add_argument("--threads", metavar="N", type=int, default=12,
                        help="CPU threads of polars and numba (default: 12)")
    args = parser.parse_args()

    from avito_ranker.threads import limit_threads

    limit_threads(args.threads)
    from avito_ranker.ensemble.config import load_ensemble_config
    from avito_ranker.paths import path, shown
    from avito_ranker.validation.gate import run_gate

    config = load_ensemble_config(args.recipe)
    logs_dir = args.logs_dir or path("outputs.member_logs")
    if args.in_dir is None:
        candidate = f"{shown(path('outputs.submissions'))} and {shown(path('outputs.predictions'))}"
    else:
        candidate = shown(args.in_dir)
    print(f"submission and pools from {candidate}", file=sys.stderr)
    print(f"reference pools from {shown(path('outputs.predictions'))}", file=sys.stderr)
    print(f"member logs from {shown(logs_dir)}", file=sys.stderr, flush=True)

    report = run_gate(config, folder=args.in_dir, logs_dir=logs_dir)
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
