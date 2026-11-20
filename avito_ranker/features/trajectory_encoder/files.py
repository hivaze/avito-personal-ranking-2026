"""Where the encoder's artefacts live, and which models exist.

A model is named ``<corpus>_<variant>``. The corpus says which history
snapshots it was trained on:

``all``       train, val and test snapshots (scores test rows; also the scale reference)
``no_test``   train and val snapshots (scores val rows, which must not benefit from test snapshots)
``window1`` .. ``window3``   train snapshots up to and inside a block of train days (scores train rows of that block)

The variant is one of two settings of the item-ID table, ``id64`` and
``id128_bias``; every input is the mean over the two.

All locations come from the ``[encoder]`` tables of ``configs/paths.toml``; the
settings of every step are in ``configs/recipes/encoder.toml`` (``recipe_file()``).
"""
from __future__ import annotations

from pathlib import Path

from avito_ranker.paths import path, recipe_dir

VARIANTS = ("id64", "id128_bias")
WINDOWS = (1, 2, 3)
SCORE_KEYS = ("fe", "new", "fe_noid", "hmax32", "hmax8", "idmax32")   # the six raw scores, see score.py


def recipe_file() -> Path:
    """The encoder recipe, ``encoder.toml`` in the registered recipe folder."""
    return recipe_dir() / "encoder.toml"


def corpus_dir(corpus: str) -> Path:
    """Folder with the token, target and count arrays of a corpus (``all`` or ``window<k>``).

    The ``no_test`` models read the ``all`` corpus and skip its test sequences.
    """
    return path(f"encoder.corpus.{'all' if corpus == 'no_test' else corpus}")


def content_files(content: str) -> tuple[Path, Path]:
    """``(table, mask)`` of a content table: int8 ``[n_items, 128]`` and bool ``[n_items]``, indexed by item id."""
    return path(f"encoder.content.{content}"), path(f"encoder.content.{content}_mask")


def checkpoint_file(model: str) -> Path:
    """The checkpoint of a model (``<corpus>_<variant>``), a dict ``{"model": state_dict}``."""
    return path(f"encoder.checkpoints.{model}")


def score_file(content: str, target: str, model: str, key: str) -> Path:
    """File with one raw score of a model on a target.

    ``target`` is ``val``, ``test``, ``window<k>`` (all train feeds of the
    block) or ``window<k>_sample`` (the 60,000-feed sample used to align
    scales). The registered location carries a ``{key}`` placeholder.
    """
    return Path(str(path(f"encoder.scores.{content}.{target}.{model}")).format(key=key))
