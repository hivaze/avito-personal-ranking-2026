"""Saving and loading ranker checkpoints.

Two layouts exist on disk. Checkpoints written by this package hold the model
config and the weights under the parameter names of
:class:`~avito_ranker.model.Ranker`. The checkpoints of the competition
members were written by the research code, whose model class had other
attribute names and many switches; :func:`_read_stored_member` translates
them. Both are read by :func:`read_checkpoint`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from avito_ranker.model.config import RankerConfig
from avito_ranker.model.inputs import InputColumn, InputView
from avito_ranker.model.ranker import RANDOM_ORDER, Ranker

FORMAT = "avito_ranker.checkpoint.v1"


@dataclass
class Checkpoint:
    """A model config, its weights and whatever the writer stored next to them."""

    config: RankerConfig
    state: dict[str, Tensor]
    inference_domain: int = RANDOM_ORDER
    info: dict = field(default_factory=dict)  # for example the training arguments and the step


def save_checkpoint(file: str | Path, model: Ranker, **info) -> None:
    """Write the config and weights of ``model``; ``info`` is stored as given (plain values only)."""
    torch.save({
        "format": FORMAT,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "inference_domain": model.inference_domain,
        "info": info,
    }, file)


def read_checkpoint(file: str | Path, device: str | torch.device = "cpu") -> Checkpoint:
    """Read a checkpoint written by :func:`save_checkpoint` or one of the stored member checkpoints."""
    raw = torch.load(file, map_location=device, weights_only=True)
    if raw.get("format") == FORMAT:
        return Checkpoint(RankerConfig.from_dict(raw["config"]), raw["state_dict"], raw["inference_domain"],
                          raw["info"])
    if "cfg" in raw:
        return _read_stored_member(raw)
    raise ValueError(f"{file}: not a ranker checkpoint")


def build_ranker(checkpoint: Checkpoint, n_finetune_extras: int = 0) -> Ranker:
    """Build a ranker with the weights of ``checkpoint`` (on the CPU).

    ``n_finetune_extras`` adds the fine-tune branch to a model trained without
    it. The branch keeps its fresh initial weights: it is drawn from the
    current random state after all other layers, and its output layer is zero.
    """
    config = checkpoint.config
    adds_branch = bool(n_finetune_extras) and not config.n_finetune_extras
    if adds_branch:
        config = config.with_finetune_extras(n_finetune_extras)
    model = Ranker(config)
    result = model.load_state_dict(checkpoint.state, strict=False)
    allowed_missing = {key for key in model.state_dict() if key.startswith("finetune_mlp.")} if adds_branch else set()
    if result.unexpected_keys or set(result.missing_keys) - allowed_missing:
        raise ValueError(f"checkpoint does not fit the model: missing {result.missing_keys}, "
                         f"unexpected {result.unexpected_keys}")
    model.inference_domain = checkpoint.inference_domain
    return model


def load_ranker(file: str | Path, device: str | torch.device = "cpu", n_finetune_extras: int = 0) -> Ranker:
    """Read a checkpoint and return its ranker on ``device``."""
    return build_ranker(read_checkpoint(file, device), n_finetune_extras).to(device)


# ---------------------------------------------------------------------------
# Stored member checkpoints
# ---------------------------------------------------------------------------
# A stored checkpoint is {"state_dict", "cfg", "args"}: the weights under the
# attribute names of the research model, the keyword arguments that built it,
# and the trainer's command line. The tables below map every old name to the
# name the same tensor has in avito_ranker.model.Ranker.

_TOP_LEVEL = {
    "f_mean": "dense_mean",                 # standardisation of the dense block
    "f_std": "dense_std",
    "cat": "category",
    "c_lin": "candidate_content",
    "c_miss": "candidate_missing_vector",
    "c_mlp": "dense_mlp",
    "c_ln": "candidate_norm",
    "retr": "retrieval",
    "dom": "domain",
    "u_lin": "user_content",
    "h_lin": "history_content",
    "h_miss": "history_missing_vector",
    "et": "event_type",
    "es": "event_source",
    "rec": "recency",
    "h_ln": "history_norm",
    "hist_enc": "history_encoder",
    "cross": "cross_blocks",
    "sets": "set_blocks",
    "head": "head",
    "aux_head": "position_head",
    "wide": "wide",
    "extra": "finetune_mlp",
}
_RETRIEVAL = {
    "wk": "key",
    "a": "similarity_scale",
    "beta": "type_bias",
    "sink": "sink",
    "e_type": "event_type",
    "e_src": "event_source",
    "e_rec": "recency",
    "t1": "difference_in",
    "t2": "difference_out",
    "v_ln": "value_norm",
    "Z": "output",
}
_BLOCK = {"ln1": "norm_query", "lnc": "norm_context", "attn": "attention", "ln2": "norm_ffn", "ff": "ffn"}
_ATTENTION = {"q": "query", "k": "key", "v": "value", "o": "output", "g": "gate", "bias_scale": "similarity_scale"}

# Switches of the research model and the value each has in every final member.
# A checkpoint with another value used a block this package does not contain.
_FIXED_SWITCHES = {
    "use_hist": True, "use_set": True, "use_pos_tower": False, "aux_pos": True, "sim_bias": 1, "domain": True,
    "intent": False, "id_buckets": 0, "arch": "tf", "xt_drop": 0.0, "attn_gate": True, "split_kv": True,
}
_KNOWN_KEYS = set(_FIXED_SWITCHES) | {
    "d", "heads", "hist_layers", "blocks", "drop", "ff", "retr_m", "n_dense", "xt_names", "xt_rel", "xt_relonly",
    # Read by no code path of the final members:
    "pos_mode", "id_dim", "id_drop", "id_hist_only", "xt_idcols",
}


def rename_stored_key(key: str) -> str:
    """Translate one parameter name of a stored member checkpoint, for example
    ``cross.0.attn.q.weight`` to ``cross_blocks.0.attention.query.weight``."""
    top, *rest = key.split(".")
    renamed = [_TOP_LEVEL[top]]
    if top == "retr":
        renamed += [_RETRIEVAL[rest[0]], *rest[1:]]
    elif top in ("hist_enc", "cross", "sets"):
        index, part, *tail = rest
        renamed += [index, _BLOCK[part]]
        if part == "attn":
            renamed += [_ATTENTION[tail[0]], *tail[1:]]
        else:
            renamed += tail
    else:
        renamed += rest
    return ".".join(renamed)


def _stored_view(names: list[str], ranked: list[int], rank_only: list[int]) -> InputView:
    """Rebuild the input view from the lists a stored checkpoint keeps.

    The research loader chose the scaling of an input from its file name; the
    rules are spelled out here once so that recipes can state the scaling.
    """
    columns = []
    for position, name in enumerate(names):
        if name.startswith("has:"):
            file, scaling = name.removeprefix("has:"), "presence"
        elif name.endswith("_rk"):
            raise ValueError(f"input {name!r}: stored rank inputs are not used by any final member")
        elif name.endswith("_z"):
            file, scaling = name, "clip_5"
        elif name.startswith(("seq_cv", "it_hist")):
            file, scaling = name, "log_count"
        else:
            file, scaling = name, "clip_10"
        columns.append(InputColumn(file, scaling, rank_and_z=position in ranked, keep_raw=position not in rank_only))
    return InputView(tuple(columns))


def _read_stored_member(raw: dict) -> Checkpoint:
    """Translate a stored member checkpoint: its ``cfg`` into a config, its weight names into ours."""
    cfg = raw["cfg"]
    unknown = set(cfg) - _KNOWN_KEYS
    wrong = {key: cfg[key] for key, value in _FIXED_SWITCHES.items() if cfg.get(key) != value}
    if unknown or wrong or not cfg.get("retr_m") or not cfg.get("xt_names"):
        raise ValueError(f"checkpoint was built with options the final members do not use: "
                         f"{sorted(unknown) or ''} {wrong or ''}")
    view = _stored_view(cfg["xt_names"], cfg.get("xt_rel") or [], cfg.get("xt_relonly") or [])
    config = RankerConfig(view=view, d_model=cfg["d"], n_heads=cfg["heads"], n_history_layers=cfg["hist_layers"],
                          n_blocks=cfg["blocks"], ffn_multiplier=cfg["ff"], dropout=cfg["drop"],
                          retrieval_slots=cfg["retr_m"])
    if config.n_dense != cfg["n_dense"]:
        raise ValueError(f"dense block of {config.n_dense} columns, checkpoint says {cfg['n_dense']}")
    state = {rename_stored_key(key): value for key, value in raw["state_dict"].items()}
    if "finetune_mlp.0.weight" in state:
        config = config.with_finetune_extras(state["finetune_mlp.0.weight"].shape[1])
    return Checkpoint(config, state, RANDOM_ORDER, {"args": raw.get("args", {})})
