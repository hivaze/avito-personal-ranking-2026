"""Model hyper-parameters and the recipe files that hold them."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from avito_ranker.data.flatten import MAX_HISTORY, N_CATEGORIES, N_EVENT_SOURCES, N_EVENT_TYPES
from avito_ranker.model.inputs import InputView
from avito_ranker.paths import recipe_dir

# Sizes fixed by the data (defined in avito_ranker.data.flatten).
EMBEDDING_DIM = 128    # content embeddings of items and of the user
__all__ = ["EMBEDDING_DIM", "MAX_HISTORY", "N_CATEGORIES", "N_COMPUTED_COLUMNS", "N_EVENT_SOURCES", "N_EVENT_TYPES",
           "RankerConfig", "load_recipe"]

# Columns of the dense block that the model computes from the raw feed
# (avito_ranker.model.dense lists them).
N_COMPUTED_COLUMNS = 63


@dataclass(frozen=True)
class RankerConfig:
    """Everything needed to build a :class:`~avito_ranker.model.Ranker`.

    ``view`` lists the pre-computed inputs. ``n_finetune_extras`` is 0 while a
    member trains on the train split and 42 during the fine-tune on the
    validation days, where it adds a branch for inputs that exist only for val
    and test rows.
    """

    view: InputView
    d_model: int = 192
    n_heads: int = 4
    n_history_layers: int = 1
    n_blocks: int = 2
    ffn_multiplier: int = 2
    dropout: float = 0.1
    retrieval_slots: int = 16
    n_finetune_extras: int = 0

    @property
    def n_dense(self) -> int:
        """Width of the dense block: 104 for the full view, 93 for the popularity-light view."""
        return N_COMPUTED_COLUMNS + self.view.n_dense_columns

    def with_finetune_extras(self, n_extras: int) -> RankerConfig:
        """The same model with the fine-tune branch for ``n_extras`` inputs."""
        return replace(self, n_finetune_extras=n_extras)

    @classmethod
    def from_dict(cls, table: dict) -> RankerConfig:
        """Build a config from the ``model`` and ``view`` tables of a recipe or a checkpoint."""
        return cls(view=InputView.from_rows(table["view"]["inputs"]), **table["model"])

    def to_dict(self) -> dict:
        """The config as plain dictionaries, in the layout of a recipe file."""
        model = {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_history_layers": self.n_history_layers,
            "n_blocks": self.n_blocks,
            "ffn_multiplier": self.ffn_multiplier,
            "dropout": self.dropout,
            "retrieval_slots": self.retrieval_slots,
            "n_finetune_extras": self.n_finetune_extras,
        }
        return {"model": model, "view": {"inputs": self.view.to_rows()}}


def load_recipe(recipe: str | Path) -> RankerConfig:
    """Read a model recipe.

    ``recipe`` is a path to a TOML file or the short name of one of the stored
    recipes: ``"full"`` reads ``configs/recipes/model_full.toml``.
    """
    file = Path(recipe)
    if file.suffix != ".toml":
        file = recipe_dir() / f"model_{recipe}.toml"
    with file.open("rb") as handle:
        return RankerConfig.from_dict(tomllib.load(handle))
