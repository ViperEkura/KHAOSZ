"""Mano manifold optimizer combined with AdamW.

Mano projects the momentum onto the tangent space of the Oblique manifold
(axis-wise tangent projection) and normalizes it, replacing the expensive
Newton-Schulz iteration in Muon with a cheaper manifold normalization.

Reference: https://arxiv.org/abs/2601.23000
"""

import math
from typing import Dict

import torch
from torch import nn, optim
from torch.optim import Optimizer

from astrai.optimization.optimizer.composite import (
    composite_state_dict,
    composite_step,
    composite_zero_grad,
    refresh_param_groups,
)
from astrai.optimization.optimizer.factory import OptimizerFactory
from astrai.optimization.optimizer.nora_nadamw import partition_optimizer_parameters


class Mano(Optimizer):
    """Manifold Normalized Optimizer for two-dimensional matrices.

    Each step alternates the projection axis (dim 0 / dim 1) to restrike the
    manifold along both rows and columns. The tangent momentum is computed
    without normalizing the parameter itself (v2 simplification) and the
    epsilon is added (not clamped) to the norm denominator.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        eps: float = 1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not 0 <= momentum <= 1:
            raise ValueError(f"Invalid momentum: {momentum}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "eps": eps,
            "steps": 0,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError(
                        f"Mano only supports 2D matrices, got shape {tuple(param.shape)}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            eps = group["eps"]
            dim = int(group["steps"] % 2)

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("Mano does not support sparse gradients")

                grad = param.grad
                state = self.state[param]
                momentum_buffer = state.get("momentum_buffer")
                if momentum_buffer is None:
                    momentum_buffer = torch.zeros_like(grad)
                momentum_buffer.mul_(momentum).add_(grad)
                update = (
                    grad.add(momentum_buffer, alpha=momentum)
                    if nesterov
                    else momentum_buffer
                )

                tangent = update - (
                    torch.sum(update * param.data, dim=dim, keepdim=True) * param.data
                )
                direction = tangent / (
                    torch.norm(tangent, p=2, dim=dim, keepdim=True) + eps
                )

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
                adjusted_lr = lr * 0.2 * math.sqrt(direction.shape[dim])
                param.add_(direction, alpha=-adjusted_lr)
                state["momentum_buffer"] = momentum_buffer

            group["steps"] += 1

        return loss


@OptimizerFactory.register("mano_adamw")
class ManoAdamW(Optimizer):
    """Mano for internal linear weights and AdamW for remaining parameters."""

    optimizer_name = "mano_adamw"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
    ):
        groups = partition_optimizer_parameters(model)
        all_params = [
            *groups.nora,
            *groups.nadamw_decay,
            *groups.nadamw_no_decay,
        ]
        if not all_params:
            raise ValueError(
                "Cannot build an optimizer for a model with no trainable parameters"
            )
        super().__init__(all_params, {})

        self.mano = (
            Mano(
                groups.nora,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
            )
            if groups.nora
            else None
        )

        adamw_groups = []
        if groups.nadamw_decay:
            adamw_groups.append(
                {"params": groups.nadamw_decay, "weight_decay": weight_decay}
            )
        if groups.nadamw_no_decay:
            adamw_groups.append({"params": groups.nadamw_no_decay, "weight_decay": 0.0})
        self.adamw = (
            optim.AdamW(
                adamw_groups,
                lr=lr,
                betas=(0.9, 0.95),
                fused=True,
            )
            if adamw_groups
            else None
        )
        self.param_groups = refresh_param_groups([self.mano, self.adamw])

    @torch.no_grad()
    def step(self, closure=None):
        return composite_step(
            [opt for opt in (self.mano, self.adamw) if opt is not None],
            closure,
        )

    def zero_grad(self, set_to_none: bool = True):
        composite_zero_grad(
            [opt for opt in (self.mano, self.adamw) if opt is not None],
            set_to_none,
        )

    def state_dict(self) -> Dict:
        return composite_state_dict({"mano": self.mano, "adamw": self.adamw})

    def load_state_dict(self, state_dict: Dict):
        if "muon" in state_dict or "nora" in state_dict:
            raise ValueError(
                "Checkpoint uses a different optimizer; select the matching "
                "--optimizer to resume it"
            )
        if "mano" not in state_dict or "adamw" not in state_dict:
            raise ValueError(
                "Checkpoint optimizer state is not compatible with mano_adamw"
            )

        saved_mano = state_dict["mano"]
        saved_adamw = state_dict["adamw"]
        if (self.mano is None) != (saved_mano is None):
            raise ValueError("Checkpoint Mano parameter groups do not match the model")
        if (self.adamw is None) != (saved_adamw is None):
            raise ValueError("Checkpoint AdamW parameter groups do not match the model")
        if self.mano is not None:
            self.mano.load_state_dict(saved_mano)
        if self.adamw is not None:
            self.adamw.load_state_dict(saved_adamw)
        self.param_groups = refresh_param_groups([self.mano, self.adamw])
