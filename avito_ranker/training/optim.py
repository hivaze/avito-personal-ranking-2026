"""Optimisers and the learning-rate schedule of a member.

Two optimisers share the parameters:

* Muon takes every hidden weight matrix, that is the weight of every linear
  layer with more than one input and more than one output (46 matrices with
  2,022,400 parameters in the full-view model of width 192). It
  orthogonalises each update; with ``adjust_lr_fn="match_rms_adamw"`` its
  steps have the size of AdamW steps, so both run with one learning rate.
* AdamW takes the rest. The embedding tables and the one-output weights of
  the score heads are matrices and get weight decay; vectors (biases, layer
  norms, the learned scalars) get none.

The schedule is warmup, constant, then a cooldown: the rate rises linearly
over the first steps, stays at its peak, and over the last part of the run
decays along ``1 - sqrt(progress)`` to 3% of the peak.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

FINAL_LR_FRACTION = 0.03    # the cooldown ends at 3% of the peak learning rate
ADAMW_BETAS = (0.9, 0.98)
MUON_MOMENTUM = 0.95


def muon_parameter_names(model: nn.Module) -> set[str]:
    """Names of the hidden weight matrices: the weights of linear layers with more than one input and output."""
    return {f"{name}.weight" for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and min(module.weight.shape) > 1}


def build_optimizers(model: nn.Module, learning_rate: float, weight_decay: float) -> list[torch.optim.Optimizer]:
    """Return ``[AdamW, Muon]`` over the parameters of ``model``; step them in this order."""
    to_muon = muon_parameter_names(model)
    rest = [(name, p) for name, p in model.named_parameters() if name not in to_muon]
    decay = [p for _, p in rest if p.ndim >= 2]
    no_decay = [p for _, p in rest if p.ndim < 2]
    adamw = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                               {"params": no_decay, "weight_decay": 0.0}],
                              lr=learning_rate, betas=ADAMW_BETAS)
    muon = torch.optim.Muon([p for name, p in model.named_parameters() if name in to_muon],
                            lr=learning_rate, weight_decay=weight_decay, momentum=MUON_MOMENTUM, nesterov=True,
                            adjust_lr_fn="match_rms_adamw")
    return [adamw, muon]


def set_learning_rate(optimizers: list[torch.optim.Optimizer], learning_rate: float) -> None:
    """Give every parameter group of every optimiser the same learning rate."""
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


@dataclass(frozen=True)
class Schedule:
    """Learning rate of each step: linear warmup, constant, then the 1-sqrt cooldown."""

    total_steps: int
    peak: float
    warmup_steps: int
    cooldown_start: int

    @classmethod
    def from_fraction(cls, total_steps: int, peak: float, warmup_steps: int, cooldown_fraction: float) -> Schedule:
        """A schedule whose cooldown covers the last ``cooldown_fraction`` of the steps."""
        return cls(total_steps, peak, warmup_steps, int(round((1 - cooldown_fraction) * total_steps)))

    def learning_rate(self, step: int) -> float:
        """Learning rate of optimiser step ``step``, counted from 0."""
        if step < self.warmup_steps:
            return self.peak * (step + 1) / self.warmup_steps
        if step < self.cooldown_start:
            return self.peak
        progress = (step - self.cooldown_start) / max(1, self.total_steps - self.cooldown_start)
        return self.peak * (FINAL_LR_FRACTION + (1 - FINAL_LR_FRACTION) * (1 - math.sqrt(progress)))

    def in_cooldown(self, step: int) -> bool:
        """Whether ``step`` lies in the cooldown."""
        return step >= self.cooldown_start

    def decay_left(self, step: int) -> float:
        """How much of the cooldown's decay is still ahead: 1 at its start, 0 at the 3% floor.

        Read from the learning rate itself, so other quantities can follow the
        learning rate down (see ``cooldown.position_weight``).
        """
        return (self.learning_rate(step) / self.peak - FINAL_LR_FRACTION) / (1 - FINAL_LR_FRACTION)
