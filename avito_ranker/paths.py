"""Where artefacts and recipes live on disk.

Every module asks this file for a location instead of spelling a path. The
locations themselves are data: they sit in ``configs/paths.toml`` (or in the
file named by the environment variable ``AVITO_PATHS``), so moving the data
means editing one file.

    from avito_ranker.paths import path
    flat_val = path("data.flat") / "val"

A relative entry resolves against the folder named by the file's ``root``
entry, which is itself relative to the folder of the paths file; a file
without ``root`` resolves against its own folder. ``configs/paths.toml`` has
``root = ".."``, the repository root, so a copy of it placed elsewhere needs
its ``root`` (or absolute entries) adjusted.

The recipes are located the same way, through the ``[configs]`` entries:
``recipe_dir()`` holds the model, training, teacher, fine-tune and encoder
recipes, ``ensemble_dir()`` one recipe per submission.

Without ``AVITO_PATHS`` the paths file is ``configs/paths.toml`` of the
checkout the package was installed from. Only an editable install (``uv
sync``) has one; with any other install, set ``AVITO_PATHS``.
"""
from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

DEFAULT_PATHS_FILE = Path(__file__).resolve().parent.parent / "configs" / "paths.toml"


@lru_cache(maxsize=None)
def _table() -> tuple[dict, Path, Path]:
    """The parsed location table, the folder relative entries resolve against, and the file itself.

    Read once per process; tests clear the cache to switch tables.
    """
    file = Path(os.environ.get("AVITO_PATHS", DEFAULT_PATHS_FILE)).resolve()
    with file.open("rb") as fh:
        table = tomllib.load(fh)
    root = table.get("root", ".")
    if not isinstance(root, str):
        raise KeyError(f"{file}: 'root' must be a single path")
    return table, (file.parent / root).resolve(), file


def paths_file() -> Path:
    """The paths file in use."""
    return _table()[2]


def root() -> Path:
    """The folder that relative entries of the paths file resolve against."""
    return _table()[1]


def path(key: str) -> Path:
    """Return the location registered under a dotted key such as ``data.flat``.

    Relative entries are resolved against :func:`root`.
    """
    node, base, file = _table()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"no path registered under {key!r} in {file}")
        node = node[part]
    if not isinstance(node, str):
        raise KeyError(f"{key!r} names a group of paths, not a single path")
    p = Path(node)
    return p if p.is_absolute() else base / p


def recipe_dir() -> Path:
    """Folder of the model, training, teacher, fine-tune and encoder recipes (``configs.recipes``)."""
    return path("configs.recipes")


def ensemble_dir() -> Path:
    """Folder of the ensemble recipes, one per submission (``configs.ensembles``)."""
    return path("configs.ensembles")


def shown(location: Path) -> str:
    """A location as messages name it: relative to :func:`root` when inside it."""
    location = Path(location)
    base = root()
    return str(location.relative_to(base)) if location.is_relative_to(base) else str(location)
