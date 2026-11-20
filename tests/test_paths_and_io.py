"""Where locations resolve, and the rules for writing over stored files."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from avito_ranker import paths
from avito_ranker.io import clear_for_rewrite, refuse_existing, save_array, save_frame
from avito_ranker.threads import limit_threads
from support import REPO


@pytest.fixture
def paths_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ``avito_ranker.paths`` at a paths file written by the test; restore the default afterwards."""
    def _use(text: str, folder: Path = tmp_path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / "paths.toml"
        file.write_text(text)
        monkeypatch.setenv("AVITO_PATHS", str(file))
        paths._table.cache_clear()
        return file

    yield _use
    paths._table.cache_clear()


def test_the_default_paths_file_resolves_against_the_repository():
    paths._table.cache_clear()
    if "AVITO_PATHS" in os.environ:
        pytest.skip("AVITO_PATHS is set in the environment")
    assert paths.paths_file() == REPO / "configs" / "paths.toml"
    assert paths.root() == REPO
    assert paths.path("data.flat") == REPO / "artifacts" / "data" / "flat"
    assert paths.recipe_dir() == REPO / "configs" / "recipes"
    assert paths.ensemble_dir() == REPO / "configs" / "ensembles"
    assert paths.shown(REPO / "artifacts" / "predictions") == "artifacts/predictions"


def test_relative_entries_resolve_against_the_paths_file(tmp_path: Path, paths_file):
    paths_file('[data]\nflat = "flat"\n[configs]\nrecipes = "recipes"\n', tmp_path / "elsewhere")
    assert paths.path("data.flat") == tmp_path / "elsewhere" / "flat"
    assert paths.recipe_dir() == tmp_path / "elsewhere" / "recipes"


def test_a_root_entry_moves_every_relative_entry(tmp_path: Path, paths_file):
    paths_file(f'root = "{REPO}"\n[data]\nflat = "artifacts/data/flat"\nraw = "/abs/raw"\n', tmp_path / "copy")
    assert paths.path("data.flat") == REPO / "artifacts" / "data" / "flat"
    assert paths.path("data.raw") == Path("/abs/raw")                  # absolute entries stay as they are
    with pytest.raises(KeyError, match="configs.recipes"):
        paths.recipe_dir()                                             # recipes are located through the file too


def test_refuse_existing_names_every_existing_file(tmp_path: Path):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    refuse_existing(a, b)
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    with pytest.raises(FileExistsError, match="a.npy.*b.npy"):
        refuse_existing(a, b)


def test_save_array_leaves_equal_values_refuses_others_and_unlinks_before_replacing(tmp_path: Path):
    stored = tmp_path / "cache" / "x.npy"
    save_array(stored, np.array([1.0, np.nan, 3.0], np.float32))
    copy = tmp_path / "copy.npy"
    try:
        os.link(stored, copy)                                          # the cache shares its files with another folder
    except OSError:
        pytest.skip("the temporary folder does not support hard links")
    inode = stored.stat().st_ino
    save_array(stored, np.array([1.0, np.nan, 3.0], np.float32))       # same values, NaN included: left alone
    assert stored.stat().st_ino == inode
    with pytest.raises(FileExistsError, match="other values"):
        save_array(stored, np.array([1.0, 2.0, 3.0], np.float32))
    with pytest.raises(FileExistsError):
        save_array(stored, np.array([1, 2, 3], np.int32))              # another dtype counts as other values
    save_array(stored, np.array([7.0, 8.0, 9.0], np.float32), overwrite=True)
    assert np.load(stored).tolist() == [7.0, 8.0, 9.0]
    assert stored.stat().st_ino != inode
    assert np.isnan(np.load(copy)[1]) and np.load(copy)[0] == 1.0      # the hard-linked copy keeps its values


def test_save_frame_follows_the_same_rules(tmp_path: Path):
    target = tmp_path / "feeds.parquet"
    frame = pl.DataFrame({"fidx": np.arange(3, dtype=np.int32), "M": np.array([5, 6, 7], np.int32)})
    save_frame(target, frame)
    before = target.stat().st_mtime_ns
    save_frame(target, frame)
    assert target.stat().st_mtime_ns == before
    other = frame.with_columns(pl.col("M") + 1)
    with pytest.raises(FileExistsError, match="another table"):
        save_frame(target, other)
    save_frame(target, other, overwrite=True)
    assert pl.read_parquet(target).equals(other)


def test_clear_for_rewrite(tmp_path: Path):
    target = tmp_path / "sub" / "proxy.txt"
    clear_for_rewrite(target, overwrite=False)                         # a new file: only the folder is made
    assert target.parent.is_dir()
    target.write_text("stored")
    with pytest.raises(FileExistsError):
        clear_for_rewrite(target, overwrite=False)
    clear_for_rewrite(target, overwrite=True)
    assert not target.exists()


def test_thread_caps_must_come_before_polars():
    with pytest.raises(RuntimeError, match="polars"):
        limit_threads(12)                                              # polars is loaded in the test process
