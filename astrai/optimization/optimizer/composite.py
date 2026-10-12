"""Helpers shared by composite optimizers."""

from typing import Any

import torch
from torch.optim import Optimizer


def composite_step(
    sub_optimizers: list[Optimizer], closure=None
) -> torch.Tensor | None:
    """Step each child optimizer and evaluate a closure once."""
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    for sub in sub_optimizers:
        sub.step()
    return loss


def composite_zero_grad(
    sub_optimizers: list[Optimizer], set_to_none: bool = True
) -> None:
    for sub in sub_optimizers:
        sub.zero_grad(set_to_none=set_to_none)


def composite_state_dict(
    named_sub_optimizers: dict[str, Optimizer | None],
) -> dict[str, Any]:
    """Serialize child optimizers while preserving empty slots."""
    return {
        name: sub.state_dict() if sub is not None else None
        for name, sub in named_sub_optimizers.items()
    }


def refresh_param_groups(sub_optimizers: list[Optimizer]) -> list[dict]:
    """Concatenate parameter groups from each non-empty child optimizer."""
    groups: list[dict] = []
    for sub in sub_optimizers:
        if sub is not None:
            groups.extend(sub.param_groups)
    return groups
