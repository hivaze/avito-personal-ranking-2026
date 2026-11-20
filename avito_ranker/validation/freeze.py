"""Freeze-day check: are the files about to be selected the files that were gated and scored.

Final submissions are selected days after they were built and scored. Before
selecting, three things are checked for every candidate file:

1. The candidate table. What the records say about the file: its members, its
   validation NDCG, its public score, its sha256. The table is the ``[record]``
   section of the recipe in ``configs/ensembles/``.
2. The sha256 of the file on disk equals the sha256 on record.
3. The gate, re-run today on the stored files, passes and prints the same text
   as the gate output stored when the file was built.

A difference in the gate text is reported and does not fail the check by
itself: the verdict line decides. A sha256 mismatch always fails.
"""
from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path

from avito_ranker.ensemble.config import EnsembleConfig, submission_file
from avito_ranker.paths import path
from avito_ranker.validation.gate import run_gate

_READ_BLOCK = 1 << 22
_DIFF_LINES_SHOWN = 20


@dataclass(frozen=True)
class FreezeResult:
    """Outcome of the freeze-day check of one candidate."""

    name: str
    sha256: str                 # of the file on disk
    sha256_matches: bool        # equal to the sha256 on record
    gate_verdict: str           # PASS, PASS with WARN or FAIL
    gate_passed: bool           # no FAIL finding
    gate_output: Path           # where the text of today's gate run was written
    gate_matches_record: bool   # today's text equals the stored gate output
    gate_diff: tuple[str, ...]  # first lines of the difference, empty when the texts are equal

    @property
    def ok(self) -> bool:
        """The file is the one on record and the gate passes today."""
        return self.sha256_matches and self.gate_passed


def file_sha256(file: Path) -> str:
    """Hex sha256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(file, "rb") as fh:
        for block in iter(lambda: fh.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_table(configs: list[EnsembleConfig]) -> list[str]:
    """One line per candidate with what the records say about it, names padded to one width (at least 10)."""
    width = max([10, *(len(config.name) for config in configs)])
    lines = []
    for config in configs:
        record = config.record
        lines.append(
            f"{config.name:{width}s} | {config.submission} | {len(config.full_view)} full-view members + "
            f"{len(config.popularity_light)} in the pool at weight {config.popularity_light_weight_text()} | "
            f"val {record.val_ndcg:.5f} | "
            f"public {record.public_ndcg} | sha256 {record.sha256}"
        )
    return lines


def recorded_gate_output(config: EnsembleConfig) -> Path:
    """Location of the gate output that was stored when the file was built."""
    return path("records.gate_outputs") / config.record.gate_output


def check_candidate(config: EnsembleConfig, out_dir: Path, logs_dir: Path | None = None) -> FreezeResult:
    """Hash the stored submission, re-run the gate on it and compare with the record.

    Today's gate text is written to ``out_dir/gate_<name>.txt``. Nothing else
    is written, and the stored gate output is only read. The gate reads the
    member logs from ``logs_dir`` (default: ``outputs.member_logs``).
    """
    sha256 = file_sha256(submission_file(config))
    report = run_gate(config, echo=False, logs_dir=logs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_output = out_dir / f"gate_{config.name}.txt"
    gate_output.write_text(report.text)
    recorded = recorded_gate_output(config).read_text()
    diff = difflib.unified_diff(recorded.splitlines(), report.text.splitlines(), "record", "today", lineterm="")
    return FreezeResult(
        name=config.name,
        sha256=sha256,
        sha256_matches=sha256 == config.record.sha256,
        gate_verdict=report.verdict,
        gate_passed=report.passed,
        gate_output=gate_output,
        gate_matches_record=report.text == recorded,
        gate_diff=tuple(list(diff)[:_DIFF_LINES_SHOWN]),
    )
