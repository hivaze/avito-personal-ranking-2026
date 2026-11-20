"""Shared pytest set-up.

Two markers (registered in ``pyproject.toml``) separate the tests that need
more than a checkout:

``artefacts``
    The test reads stored data (predictions, submissions, flat arrays). It
    asks for the files with the ``need`` fixture and is skipped when one is
    absent.
``gpu``
    The test needs a CUDA device and is skipped without one.

``pytest -m "not artefacts and not gpu"`` runs the rest anywhere.

The ``tiny_repo`` fixture builds a small synthetic copy of the artefact tree
(two splits, member predictions, pools, logs, a sample submission) in a
temporary folder and points ``avito_ranker.paths`` at it, so the ensemble and
the gate can run end to end without the competition data.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from avito_ranker import paths
from avito_ranker.validation import calibration, private_mix
from support import CONFIGS_TABLE


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if not gpu_items:
        return
    import torch

    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="no CUDA device")
        for item in gpu_items:
            item.add_marker(skip)


@pytest.fixture
def need():
    """Return a function that skips the test unless all the given files exist."""
    def _need(*files: Path) -> None:
        missing = [str(f) for f in files if not Path(f).exists()]
        if missing:
            pytest.skip(f"artefact absent: {missing[0]}" + (f" (+{len(missing) - 1} more)" if len(missing) > 1 else ""))

    return _need


def _clear_path_caches() -> None:
    """Forget everything that was computed for another artefact tree."""
    paths._table.cache_clear()
    calibration.size_weights.cache_clear()
    private_mix._items_seen_in_train.cache_clear()
    private_mix.cold_share.cache_clear()
    private_mix.mix_weights.cache_clear()


# The synthetic tree. Val covers Tuesday and Thursday of the real validation
# week (Moscow time) so that the gate's day slices are populated, and test
# straddles the public / private boundary. Feed sizes and cold shares cycle
# through every cell of the private-mix grid, so each test cell has val support.
TUESDAY_NOON = 20718 * 86400 + 9 * 3600
THURSDAY_NOON = 20720 * 86400 + 9 * 3600
SIZE_RANGES = ((12, 26), (26, 51), (51, 61))   # one per feed-size bin: 11-25, 26-50, 51-100 candidates
COLD_SHARES = (0.10, 0.20, 0.40, 0.60)         # one per cold-share bin: <15%, 15-30%, 30-45%, above 45%
FINETUNE_TAG = "ft1"
FULL_VIEW = ("full_a", "full_b", "full_c")
POPULARITY_LIGHT = ("light_a", "light_b")
SAFE_INPUTS = ("seq_cv_pop", "ctx_nd_max", "pre2_fe_s", "has:pre2_fe")


@dataclass(frozen=True)
class TinyRepo:
    """Locations inside the synthetic artefact tree and the arrays it was built from."""

    root: Path
    config_path: Path
    val_sizes: np.ndarray
    test_sizes: np.ndarray
    val_labels: np.ndarray
    val_cold_share: np.ndarray    # per val feed, by construction
    test_cold_share: np.ndarray   # per test feed, by construction

    @property
    def predictions(self) -> Path:
        return self.root / "preds"

    @property
    def submissions(self) -> Path:
        return self.root / "subs"

    @property
    def member_logs(self) -> Path:
        return self.root / "logs"

    @property
    def flat(self) -> Path:
        return self.root / "flat"

    def write_member_log(self, member: str, inputs: tuple[str, ...], file_name: str | None = None) -> None:
        """Append a training run with the two lines the gate reads to a log file."""
        with open(self.member_logs / (file_name or f"{member}.log"), "a") as log:
            log.write(f"==== run args: {{'variant': '{member}', 'seed': 1}}\n"
                      f"train-stage extra features ({len(inputs)}): {list(inputs)}\n"
                      "step 100 loss 1.0\n")

    def write_finetune_log(self, member: str, inputs: tuple[str, ...]) -> None:
        """Write the member's fine-tune log with the line the gate reads."""
        (self.member_logs / f"{member}_{FINETUNE_TAG}_ft.log").write_text(
            f"==== run args: {{'variant': '{member}', 'out': '{member}_{FINETUNE_TAG}'}}\n"
            f"extra features ({len(inputs)}): {list(inputs)}\n")


def _feed_sizes(rng: np.random.Generator, n_feeds: int) -> np.ndarray:
    """Feed sizes that visit the three size ranges in turn, four feeds at a time."""
    return np.array([rng.integers(*SIZE_RANGES[(i // 4) % 3]) for i in range(n_feeds)], np.int64)


def _cold_rows(sizes: np.ndarray) -> np.ndarray:
    """Mark the first rows of each feed as cold so that feed ``i`` has the cold share ``COLD_SHARES[i % 4]``."""
    cold = np.zeros(int(sizes.sum()), bool)
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    for i, (offset, size) in enumerate(zip(offsets, sizes)):
        cold[offset:offset + round(COLD_SHARES[i % 4] * size)] = True
    return cold


def _feed_mean(per_row: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Mean of a per-row array inside each feed."""
    return np.add.reduceat(per_row.astype(np.float64), np.r_[0, np.cumsum(sizes)[:-1]]) / sizes


def _write_split(folder: Path, sizes: np.ndarray, users: np.ndarray, timestamps: np.ndarray, first_item: int,
                 first_feed_id: int) -> np.ndarray:
    """Write the feed table and the candidate arrays of one split; return the item ids."""
    folder.mkdir(parents=True)
    n_feeds, n_rows = len(sizes), int(sizes.sum())
    offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    pl.DataFrame({
        "fidx": np.arange(n_feeds, dtype=np.int32),
        "feed_id": (first_feed_id + np.arange(n_feeds)).astype(np.int32),
        "timestamp": timestamps.astype(np.int32),
        "user_id": users.astype(np.int32),
        "M": sizes.astype(np.int32),
        "H": np.zeros(n_feeds, np.int32),
        "cand_off": offsets,
        "hist_off": np.zeros(n_feeds, np.int64),
    }).write_parquet(folder / "feeds.parquet")
    items = (first_item + np.arange(n_rows)).astype(np.int32)
    np.save(folder / "cand_fidx.npy", np.repeat(np.arange(n_feeds, dtype=np.int32), sizes))
    np.save(folder / "cand_item_id.npy", items)
    return items


@pytest.fixture
def tiny_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TinyRepo]:
    """Build the synthetic artefact tree and make ``avito_ranker.paths`` resolve into it."""
    rng = np.random.default_rng(7)
    root = tmp_path / "repo"
    for sub in ("raw", "preds", "subs", "logs", "runs", "records", "quarantine",
                "features/val", "features/test", "features/train", "feature_cache/val"):
        (root / sub).mkdir(parents=True)

    # Feeds. Users 1..40 recur, so some val feeds have a later feed; user 0 is the guest.
    n_val, n_test = 240, 160
    val_sizes, test_sizes = _feed_sizes(rng, n_val), _feed_sizes(rng, n_test)
    val_users = rng.integers(0, 41, n_val)
    test_users = rng.integers(0, 41, n_test)
    val_times = np.sort(np.where(rng.random(n_val) < 0.5, TUESDAY_NOON, THURSDAY_NOON) + rng.integers(0, 3600, n_val))
    test_times = np.sort(calibration.PUBLIC_END + rng.integers(-40_000, 40_000, n_test))
    val_items = _write_split(root / "flat/val", val_sizes, val_users, val_times, first_item=1_000, first_feed_id=10)
    test_items = _write_split(root / "flat/test", test_sizes, test_users, test_times, first_item=50_000,
                              first_feed_id=5_000)

    # An item is warm if it occurs in train, as a candidate (the val items here) or in a history (the test items).
    (root / "flat/train").mkdir()
    val_cold, test_cold = _cold_rows(val_sizes), _cold_rows(test_sizes)
    np.save(root / "flat/train/cand_item_id.npy", val_items[~val_cold])
    np.save(root / "flat/train/hist_item_id.npy", test_items[~test_cold])

    # Labels: one click per val feed, and member predictions that know it a little.
    val_labels = np.zeros(int(val_sizes.sum()), np.int8)
    val_offsets = np.r_[0, np.cumsum(val_sizes)[:-1]]
    val_labels[val_offsets + rng.integers(0, val_sizes)] = 1
    np.save(root / "flat/val/cand_target.npy", val_labels)
    test_signal = rng.normal(size=int(test_sizes.sum()))
    for member in FULL_VIEW + POPULARITY_LIGHT:
        for suffix, signal in ((f"{FINETUNE_TAG}_val_oof", val_labels), ("trainonly_val", val_labels),
                               (f"{FINETUNE_TAG}_test", test_signal), ("trainonly_test", test_signal)):
            scores = 0.7 * signal + rng.normal(size=len(signal))
            np.save(root / "preds" / f"nn_{member}_{suffix}.npy", scores.astype(np.float32))

    # The reference submission's pools: weaker than the members.
    for branch in ("ft", "to"):
        np.save(root / "preds" / f"ens_REF_{branch}_val.npy",
                (0.4 * val_labels + rng.normal(size=len(val_labels))).astype(np.float32))
        np.save(root / "preds" / f"ens_REF_{branch}_test.npy",
                (0.4 * test_signal + rng.normal(size=len(test_signal))).astype(np.float32))

    # The organisers' sample submission lists the test keys in its own order.
    test_feed_ids = np.repeat(5_000 + np.arange(n_test), test_sizes).astype(np.int32)
    shuffle = rng.permutation(len(test_items))
    pl.DataFrame({"feed_id": test_feed_ids[shuffle], "item_id": test_items[shuffle],
                  "score": np.zeros(len(test_items), np.float32)}).write_parquet(root / "raw/sample_submission.parquet")

    locations = {
        "data": {"raw": "raw", "flat": "flat", "features": "features", "feature_cache": "feature_cache",
                 "quarantine": "quarantine"},
        "outputs": {"predictions": "preds", "submissions": "subs", "runs": "runs", "member_logs": "logs"},
        "records": {"gate_outputs": "records"},
    }
    toml_lines = []
    for group, entries in locations.items():
        toml_lines.append(f"[{group}]")
        toml_lines += [f'{key} = "{root / folder}"' for key, folder in entries.items()]
    (root / "paths.toml").write_text("\n".join(toml_lines) + "\n" + CONFIGS_TABLE)

    config_path = root / "tiny.toml"
    config_path.write_text("\n".join([
        'name = "TINY"',
        "[members]",
        f"full_view = {json.dumps(FULL_VIEW)}",
        f"popularity_light = {json.dumps(POPULARITY_LIGHT)}",
        "[weights]",
        "popularity_light = 0.2",
        "finetuned_branch = 0.5",
        "[files]",
        f'finetune_tag = "{FINETUNE_TAG}"',
        'pool_tag = "TINY"',
        'submission = "tiny.parquet"',
        "[gate]",
        'reference_pool_tag = "REF"',
        "[record]",
        'sha256 = "unset"',
        "val_ndcg = 0.5",
        "public_ndcg = 0.4",
        'gate_output = "gate_TINY.txt"',
    ]) + "\n")

    monkeypatch.setenv("AVITO_PATHS", str(root / "paths.toml"))
    _clear_path_caches()
    repo = TinyRepo(
        root, config_path, val_sizes, test_sizes, val_labels,
        val_cold_share=_feed_mean(val_cold, val_sizes),
        test_cold_share=_feed_mean(test_cold, test_sizes),
    )
    for member in FULL_VIEW + POPULARITY_LIGHT:
        repo.write_member_log(member, SAFE_INPUTS)
        repo.write_finetune_log(member, ("centrality", "seq_top3"))
    yield repo
    _clear_path_caches()
