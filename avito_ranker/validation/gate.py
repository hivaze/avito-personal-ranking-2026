"""The pre-submission gate: checks every submission file had to pass before it was sent.

A FAIL blocks the file. A WARN needs a written justification. The checks:

A  File validity. The (feed_id, item_id) keys equal the sample submission, no
   key twice, all scores finite, no two candidates of a feed with one score.
B  Reproduction. The order inside every test feed equals the order of the
   final score recomputed from the stored pool files (per-feed Spearman of at
   least 0.9999). Catches a file built from other predictions than declared.
   A correct file can read just below 1, for example 0.999998: the writer
   orders exact ties of the score by its random key, this check by row order.
C  Inputs audit. The inputs of every member, as the trainer and the fine-tune
   logged them, are matched against lists of banned and suspicious names, and
   no input may be a hard link to a quarantined file. The banned names are the
   constructions that leaked: statistics that include a user's own later
   snapshots, and encoder scores whose train rows were built differently from
   the val and test rows. A member whose input list is not in the logs fails.
D  Validation. Paired difference to a reference submission with its standard
   error, on all feeds and on slices. The leak probe compares the gain on the
   feeds whose user has a later feed in val or test with the gain on the other
   feeds: a leak through later snapshots concentrates on the first group.
E  Forecast of the public and private score (``calibration``). Information.
F  Parity. Agreement with the reference on test against agreement on val. A
   file that moves test rankings much more than val rankings was probably
   built from test inputs that differ from the val inputs.

The text the gate prints is kept as it was when the stored submissions were
gated, so a re-run can be compared with the stored output line by line.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import polars as pl

from avito_ranker.data.flatten import moscow_day, read_flat
from avito_ranker.ensemble.build import Branches, final_score
from avito_ranker.ensemble.config import BRANCHES, EnsembleConfig, pool_file, submission_file
from avito_ranker.ensemble.feeds import FeedIndex, load_feed_index
from avito_ranker.metrics import ndcg_per_feed
from avito_ranker.paths import path, shown
from avito_ranker.validation.calibration import predict_board

# Never allowed as an input of any stage: statistics that include the same
# user's later snapshots (docs/02-what-the-data-told-us.md), under every name they had.
BANNED_ANYWHERE = [
    r"^it_hist_(all|et\d+)(_rk|_z|_gap)?$",  # history counts without the leave-user-out subtraction
    r"^lh_",                                 # later-history flags: the item is in a later snapshot of the same user
    r"later",                                # any input named after a later snapshot
]
# Never allowed as an input of the training stage: encoder scores whose train
# rows leak or are built differently from the val and test rows.
BANNED_IN_TRAINING = [r"^pre_(fe|new|hmax|idmax|fe_noid|id_contrib)", r"_id_contrib"]
# Raw encoder scores hit the +-10 clip of the input standardiser; the rescaled copies must be used.
WARN_IN_TRAINING = [
    r"^prew_fe$", r"^prew_new$",   # prew_*: the first leak-free train build of the encoder scores, unscaled
    r"^pre2_fe$", r"^pre2_new$",   # pre2_*: the rebuild for all splits, raw
]
# Prone to drift or leakage (train-candidate statistics, ALS factors, hour of day).
WARN_ANYWHERE = [r"^als_", r"^hour$", r"^it_tr_", r"^u_tr_", r"^it_cand_cnt_train$"]

MIN_REPRODUCTION_SPEARMAN = 0.9999
SHORT_FEED = 50                      # feeds with fewer candidates form the "M<50" slice
MIN_SLICE_FEEDS = 50                 # smaller slices are not printed
THURSDAY = 20720                     # days since 1970-01-01 of 24 Sep 2026; val is Tue 22 and Thu 24 Sep
PRESENCE_PREFIX = "has:"             # "has:<input>" is the presence flag of <input>


class GateReport:
    """The lines the gate prints, with the count of FAIL and WARN findings."""

    def __init__(self, echo: bool = True):
        self.lines: list[str] = []
        self.n_fail = 0
        self.n_warn = 0
        self._echo = echo

    def say(self, line: str) -> None:
        """Add a line to the report (and print it when echoing)."""
        self.lines.append(line)
        if self._echo:
            print(line, flush=True)

    def fail(self, message: str) -> None:
        """Record a finding that blocks the file."""
        self.n_fail += 1
        self.say(f"[FAIL] {message}")

    def warn(self, message: str) -> None:
        """Record a finding that needs a written justification."""
        self.n_warn += 1
        self.say(f"[WARN] {message}")

    @property
    def verdict(self) -> str:
        """``FAIL``, ``PASS with WARN`` or ``PASS``."""
        if self.n_fail:
            return "FAIL"
        return "PASS with WARN" if self.n_warn else "PASS"

    @property
    def passed(self) -> bool:
        """True when no check failed (warnings allowed)."""
        return self.n_fail == 0

    @property
    def text(self) -> str:
        """The whole report as printed, one line per finding or read-out."""
        return "\n".join(self.lines) + "\n"


class MemberLogs:
    """Reads the input lists that training and fine-tune runs wrote into their logs.

    A training log holds the run's arguments on a line with ``args: {``,
    followed by ``train-stage extra features (N): [...]``. A log is appended
    to, so one file may hold several runs, and a run that crashed may be
    followed by a re-run under the same name; the inputs of the last run that
    logged them are the ones audited. Files are read in the order ``*.out``,
    then ``*.log``, each sorted by name. A fine-tune log is named
    ``<member>_<finetune tag>_ft.log`` and lists its inputs on the line
    ``extra features (N): [...]``; the last such line counts.
    """

    def __init__(self, folder: Path | None = None):
        self.folder = Path(folder) if folder is not None else path("outputs.member_logs")
        self._files = sorted(self.folder.glob("*.out")) + sorted(self.folder.glob("*.log"))
        self._headers: dict[Path, list[str]] = {}

    def _header_lines(self, log: Path) -> list[str]:
        """The lines of ``log`` that start a run or list its inputs."""
        if log not in self._headers:
            lines = log.read_text(errors="ignore").splitlines()
            self._headers[log] = [
                line for line in lines if "args: {" in line or line.startswith("train-stage extra features")
            ]
        return self._headers[log]

    def training_inputs(self, member: str) -> list[str] | None:
        """Inputs of the last training run named ``member``, or None if no log lists them."""
        marker = f"'variant': '{member}'"
        found = None
        for log in self._files:
            in_run = False
            for line in self._header_lines(log):
                if "args: {" in line:
                    in_run = marker in line
                elif in_run:
                    found = _quoted_names(line)
                    in_run = False
        return found

    @property
    def shown_folder(self) -> str:
        """The logs folder as messages name it: relative to the paths root when inside it."""
        return shown(self.folder)

    @staticmethod
    def finetune_log_name(member: str, finetune_tag: str) -> str:
        """File name of a member's fine-tune log."""
        return f"{member}_{finetune_tag}_ft.log"

    def finetune_inputs(self, member: str, finetune_tag: str) -> list[str] | None:
        """Extra inputs of the member's last fine-tune, or None if its log is missing or lists none."""
        log = self.folder / self.finetune_log_name(member, finetune_tag)
        if not log.exists():
            return None
        found = None
        for line in log.read_text(errors="ignore").splitlines():
            if line.startswith("extra features"):
                found = _quoted_names(line)
        return found


def _quoted_names(line: str) -> list[str]:
    """The quoted names of a logged Python list: ``label (3): ['a', 'b', 'c']``."""
    return re.findall(r"'([^']+)'", line.split(":", 1)[1])


def _matches(name: str, patterns: list[str]) -> bool:
    """Whether ``name`` matches one of the regular expressions."""
    return any(re.search(p, name) for p in patterns)


def _standard_error(x: np.ndarray) -> float:
    """Standard error of the mean of ``x``."""
    return x.std() / np.sqrt(len(x))


def within_feed_rank(feeds: FeedIndex, x: np.ndarray) -> np.ndarray:
    """Rank of every row inside its feed, 0 = lowest; equal values keep their row order."""
    order = np.lexsort((x, feeds.feed_of_row))
    rank = np.empty(len(x), np.int64)
    rank[order] = feeds.row_in_feed
    return rank


def per_feed_spearman(feeds: FeedIndex, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation of two score columns inside every feed (1 where a column is constant)."""
    feed = feeds.feed_of_row

    def centred_rank(x: np.ndarray) -> np.ndarray:
        """Within-feed rank minus its feed mean."""
        rank = within_feed_rank(feeds, x).astype(np.float64)
        return rank - (np.bincount(feed, rank) / feeds.sizes)[feed]

    ca, cb = centred_rank(a), centred_rank(b)
    covariance = np.bincount(feed, ca * cb)
    scale = np.sqrt(np.bincount(feed, ca * ca) * np.bincount(feed, cb * cb))
    return np.where(scale > 0, covariance / np.maximum(scale, 1e-12), 1.0)


def _stored_final_score(report: GateReport, pool_tag: str, split: str, feeds: FeedIndex,
                        finetuned_weight: float, folder: Path | None = None) -> np.ndarray:
    """Final score of an ensemble on ``split``, recomputed from its stored pool files."""
    columns = []
    for branch in BRANCHES:
        file = pool_file(pool_tag, branch, split, folder)
        x = np.load(file).astype(np.float32)
        if not np.isfinite(x).all():
            report.fail(f"{file.stem}: non-finite predictions")
        columns.append(np.nan_to_num(x))
    return final_score(Branches(*columns), feeds, finetuned_weight)


def check_file(report: GateReport, submission: pl.DataFrame) -> None:
    """Check A: the file has the sample submission's keys, finite scores and no ties."""
    sample = pl.read_parquet(path("data.raw") / "sample_submission.parquet")
    joined = sample.select("feed_id", "item_id").join(submission, on=["feed_id", "item_id"], how="left")
    n_missing = joined["score"].null_count()
    if submission.height != sample.height or n_missing > 0:
        report.fail(f"A: key set differs from sample_submission ({submission.height} rows, {n_missing} missing)")
    if submission.select("feed_id", "item_id").is_duplicated().any():
        report.fail("A: duplicate keys")
    if not np.isfinite(submission["score"].to_numpy()).all():
        report.fail("A: non-finite scores")
    tied_feeds = submission.group_by("feed_id").agg((pl.col("score").n_unique() != pl.len()).alias("t"))["t"].sum()
    if tied_feeds:
        report.fail(f"A: {tied_feeds} feeds with tied scores")
    report.say("A file validity checked")


def check_reproduction(report: GateReport, submission: pl.DataFrame, test_score: np.ndarray,
                       test: FeedIndex) -> None:
    """Check B: the file orders every test feed as the recomputed final score does."""
    flat_rows = pl.DataFrame({
        "feed_id": np.repeat(test.table["feed_id"].to_numpy(), test.sizes),
        "item_id": read_flat("test", "cand_item_id", mmap=False),
        "row": np.arange(len(test_score)),
    })
    file_score = (flat_rows.join(submission, on=["feed_id", "item_id"], how="left")
                  .sort("row")["score"].to_numpy().astype(np.float64))
    rho = per_feed_spearman(test, file_score, test_score)
    if rho.min() < MIN_REPRODUCTION_SPEARMAN:
        report.fail(f"B: submission order does not match the stored pools (min per-feed Spearman {rho.min():.6f})")
    report.say(f"B reproduction: min per-feed Spearman {rho.min():.6f}")


def _quarantined_inodes() -> set[int]:
    """Inodes of the quarantined input files; a hard link to one of them is the same data."""
    return {os.stat(p).st_ino for p in path("data.quarantine").rglob("*.npy")}


def _audit_training_inputs(report: GateReport, member: str, inputs: list[str], quarantined: set[int]) -> None:
    """Banned and warn-listed names among a member's training inputs, and hard links into the quarantine."""
    input_folders = (path("data.feature_cache"), path("data.features"))
    for name in inputs:
        is_flag = name.startswith(PRESENCE_PREFIX)
        column = name[len(PRESENCE_PREFIX):] if is_flag else name
        if _matches(column, BANNED_ANYWHERE + BANNED_IN_TRAINING):
            report.fail(f"{member}: banned train-stage input {name}")
        elif _matches(column, WARN_ANYWHERE) or (not is_flag and _matches(column, WARN_IN_TRAINING)):
            report.warn(f"{member}: train-stage input {name} is on the warn list")
        for split in ("train", "val", "test"):
            for folder in input_folders:
                file = folder / split / f"{column}.npy"
                if file.exists() and os.stat(file).st_ino in quarantined:
                    report.fail(f"{member}: input {column} ({split}) is hard-linked to a quarantined file")
                    break


def _audit_finetune_inputs(report: GateReport, member: str, inputs: list[str]) -> None:
    """Banned and warn-listed names among a member's fine-tune extras."""
    for name in inputs:
        if _matches(name, BANNED_ANYWHERE):
            report.fail(f"{member}: banned fine-tune input {name}")
        elif _matches(name, WARN_ANYWHERE):
            report.warn(f"{member}: fine-tune input {name} is on the warn list")


def audit_inputs(report: GateReport, config: EnsembleConfig, logs_dir: Path | None = None) -> None:
    """Check C: every member's logged inputs against the banned and warn lists and the quarantine.

    The logs are read from ``logs_dir``, by default the registered
    ``outputs.member_logs``. A member whose train-stage or fine-tune input
    list cannot be found fails the check: inputs that cannot be read cannot
    be audited.
    """
    logs = MemberLogs(logs_dir)
    quarantined = _quarantined_inodes()
    for member in config.members:
        training_inputs = logs.training_inputs(member)
        if training_inputs is None:
            report.fail(f"C: {member}: no train-stage input list in {logs.shown_folder}; "
                        "the inputs cannot be audited")
        else:
            _audit_training_inputs(report, member, training_inputs, quarantined)
        finetune_inputs = logs.finetune_inputs(member, config.finetune_tag)
        if finetune_inputs is None:
            report.fail(f"C: {member}: no fine-tune input list in "
                        f"{logs.shown_folder}/{logs.finetune_log_name(member, config.finetune_tag)}; "
                        "the inputs cannot be audited")
        else:
            _audit_finetune_inputs(report, member, finetune_inputs)
        if training_inputs is not None and finetune_inputs is not None:
            report.say(f"  {member}: {len(training_inputs)} train-stage inputs audited, "
                       f"fine-tune {config.finetune_tag} audited")


def users_with_later_feed(val: FeedIndex, test: FeedIndex) -> np.ndarray:
    """True for val feeds whose user has a later feed in val or test.

    User 0 is the pooled guest user and never counts: its feeds belong to
    different people.
    """
    users, times = val.table["user_id"].to_numpy(), val.table["timestamp"].to_numpy()
    all_users = np.concatenate([users, test.table["user_id"].to_numpy()])
    all_times = np.concatenate([times, test.table["timestamp"].to_numpy()])
    last_feed: dict[int, int] = {}
    for user, time in zip(all_users.tolist(), all_times.tolist()):
        if time > last_feed.get(user, -1):
            last_feed[user] = time
    return np.array([(u != 0) and (last_feed[u] > t) for u, t in zip(users.tolist(), times.tolist())])


def compare_on_val(report: GateReport, candidate: np.ndarray, reference: np.ndarray,
                   val: FeedIndex, test: FeedIndex) -> None:
    """Check D: paired per-feed difference to the reference, its slices and the leak probe."""
    delta = candidate - reference
    report.say(f"D val new {candidate.mean():.5f} ref {reference.mean():.5f} "
               f"delta {delta.mean():+.5f} +- {_standard_error(delta):.5f}")
    users, times = val.table["user_id"].to_numpy(), val.table["timestamp"].to_numpy()
    has_later = users_with_later_feed(val, test)
    tuesday = moscow_day(times) < THURSDAY
    slices = {
        "M<50": val.sizes < SHORT_FEED, "M>=50": val.sizes >= SHORT_FEED,
        "Tue": tuesday, "Thu": ~tuesday, "user0": users == 0,
        "has_later": has_later, "no_later": ~has_later,
    }
    for name, mask in slices.items():
        if mask.sum() < MIN_SLICE_FEEDS:
            continue
        mean, se = delta[mask].mean(), _standard_error(delta[mask])
        report.say(f"   {name:9s} n={mask.sum():6d} delta {mean:+.5f} +- {se:.5f}")
        if mean / max(se, 1e-12) < -2:
            report.warn(f"D slice {name} below -2 SE")

    # Only 117 val feeds of the competition data have a later feed, so the probe
    # is noisy: files of one recipe differed by 2 standard errors in either
    # direction. FAIL therefore needs more than 3. Between 2 and 3 the answer
    # is a controlled check: a paired run that differs only in the suspected
    # input.
    gap = delta[has_later].mean() - delta[~has_later].mean()
    gap_se = np.sqrt(_standard_error(delta[has_later]) ** 2 + _standard_error(delta[~has_later]) ** 2)
    report.say(f"   LEAK PROBE has_later - no_later gain: {gap:+.5f} +- {gap_se:.5f} "
               "(low power on val: few has-later feeds)")
    if gap > 3 * gap_se:
        report.fail("D leak probe: the gain concentrates on feeds whose user has a later snapshot (z > 3)")
    elif gap > 2 * gap_se:
        report.warn("D leak probe: gain larger on has-later feeds (2 < z <= 3): run the controlled check")


def check_parity(report: GateReport, val: FeedIndex, test: FeedIndex,
                 val_scores: tuple[np.ndarray, np.ndarray], test_scores: tuple[np.ndarray, np.ndarray]) -> None:
    """Check F: the change against the reference moves test rankings about as much as val rankings."""
    on_val = per_feed_spearman(val, *val_scores)
    on_test = per_feed_spearman(test, *test_scores)
    report.say(f"F new-vs-ref per-feed Spearman: val mean {on_val.mean():.4f} p5 {np.percentile(on_val, 5):.4f} "
               f"| test mean {on_test.mean():.4f} p5 {np.percentile(on_test, 5):.4f}")
    if on_test.mean() < on_val.mean() - 0.02:
        report.warn("F the change moves test rankings much more than val rankings "
                    "(possible test-side construction break)")


def run_gate(config: EnsembleConfig, folder: Path | None = None, echo: bool = True,
             logs_dir: Path | None = None) -> GateReport:
    """Run checks A to F on the submission of ``config`` and return the report.

    The submission and its pool files are read from ``folder`` when given (a
    rebuild made next to the stored files), otherwise from the registered
    submissions and predictions folders. The reference pools are always the
    stored ones. The member logs are read from ``logs_dir``, by default the
    registered ``outputs.member_logs``.
    """
    report = GateReport(echo)
    val, test = load_feed_index("val"), load_feed_index("test")
    labels = read_flat("val", "cand_target", mmap=False)
    weight = config.finetuned_weight

    submission = pl.read_parquet(submission_file(config, folder))
    check_file(report, submission)

    test_new = _stored_final_score(report, config.pool_tag, "test", test, weight, folder)
    check_reproduction(report, submission, test_new, test)

    audit_inputs(report, config, logs_dir)

    val_new = _stored_final_score(report, config.pool_tag, "val", val, weight, folder)
    val_ref = _stored_final_score(report, config.reference_pool_tag, "val", val, weight)
    ndcg_new = ndcg_per_feed(val_new, labels, val.offsets, val.sizes)
    ndcg_ref = ndcg_per_feed(val_ref, labels, val.offsets, val.sizes)
    compare_on_val(report, ndcg_new, ndcg_ref, val, test)

    forecast_new, forecast_ref = predict_board(ndcg_new), predict_board(ndcg_ref)
    report.say(f"E predicted public {forecast_new['public']:.4f} private {forecast_new['private']:.4f} (+-0.003); "
               f"ref {forecast_ref['public']:.4f} / {forecast_ref['private']:.4f}")

    test_ref = _stored_final_score(report, config.reference_pool_tag, "test", test, weight)
    check_parity(report, val, test, (val_new, val_ref), (test_new, test_ref))

    report.say(f"\nGATE: {report.verdict}  (FAIL {report.n_fail}, WARN {report.n_warn})")
    return report
