"""Locations the tests share: the recipe folders of this checkout, for synthetic paths files.

A test that points ``AVITO_PATHS`` at a paths file of its own appends
``CONFIGS_TABLE`` to it, so that recipes are still found in the checkout.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECIPE_DIR = REPO / "configs" / "recipes"
ENSEMBLE_DIR = REPO / "configs" / "ensembles"
CONFIGS_TABLE = f'[configs]\nrecipes = "{RECIPE_DIR}"\nensembles = "{ENSEMBLE_DIR}"\n'
