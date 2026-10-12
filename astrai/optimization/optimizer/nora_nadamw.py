"""Nora matrix optimizer combined with Nesterov AdamW."""

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Shard
from torch.optim import Optimizer

from astrai.model.components.embedding import Embedding
from astrai.model.components.linear import Linear
from astrai.model.components.lora import LoRALinear
from astrai.model.components.norm import RMSNorm
from astrai.optimization.optimizer.composite import (
    composite_state_dict,
    composite_step,
    composite_zero_grad,
    refresh_param_groups,
)
from astrai.optimization.optimizer.factory import OptimizerFactory

NORA_EPS = 1e-10


def _row_normalize(tensor: Tensor, eps: float) -> Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=eps)


def nora_direction(update: Tensor, param: Tensor, eps: float = NORA_EPS) -> Tensor:
    """Project an update onto each parameter row's tangent space and normalize."""
    theta_hat = _row_normalize(param.to(torch.float32), eps)
    update_fp32 = update.to(torch.float32)
    radial = (update_fp32 * theta_hat).sum(dim=-1, keepdim=True) * theta_hat
    direction = _row_normalize(update_fp32 - radial, eps)
    return direction.to(update.dtype)


def nora_lr_scale(lr: float, shape: torch.Size) -> float:
    """Scale Nora's LR for tall ``[d_out, d_in]`` linear weights."""
    return lr * math.sqrt(max(1.0, shape[-2] / shape[-1]))


def _validate_complete_rows(param: Tensor) -> None:
    if not isinstance(param, DTensor):
        return
    last_dim = param.ndim - 1
    for placement in param.placements:
        if isinstance(placement, Shard) and placement.dim % param.ndim == last_dim:
            raise ValueError(
                "Nora requires complete parameter rows, but this DTensor is sharded "
                "along its last dimension"
            )


class Nora(Optimizer):
    """Normalized Orthogonal Row Alignment for two-dimensional matrices."""

    def __init__(
        self,
        params,
        lr: float = 5e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        beta: float = 0.95,
        nesterov: bool = True,
        eps: float = NORA_EPS,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not 0 <= momentum <= 1:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0 <= beta < 1:
            raise ValueError(f"Invalid beta: {beta}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "beta": beta,
            "nesterov": nesterov,
            "eps": eps,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError(
                        f"Nora only supports 2D matrices, got shape {tuple(param.shape)}"
                    )
                _validate_complete_rows(param)

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
            beta = group["beta"]
            nesterov = group["nesterov"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("Nora does not support sparse gradients")

                grad = param.grad
                state = self.state[param]
                momentum_buffer = state.get("momentum_buffer")
                if momentum_buffer is None:
                    momentum_buffer = torch.zeros_like(grad)
                momentum_buffer.lerp_(grad, 1 - beta)
                update = (
                    grad.lerp(momentum_buffer, momentum)
                    if nesterov
                    else momentum_buffer
                )
                direction = nora_direction(update, param, eps)

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
                param.add_(direction, alpha=-nora_lr_scale(lr, param.shape))
                state["momentum_buffer"] = momentum_buffer

        return loss


class NAdamW(Optimizer):
    """AdamW using the reference Nesterov first-moment update."""

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        beta1, beta2 = betas
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("NAdamW does not support sparse gradients")

                grad = param.grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                state["step"] += 1
                first_moment = state["m"]
                second_moment = state["v"]
                first_moment.mul_(beta1).add_(grad, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                nesterov_moment = (
                    beta1 * first_moment + (1 - beta1) * grad
                ) / bias_correction1
                corrected_second_moment = second_moment / bias_correction2

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
                param.addcdiv_(
                    nesterov_moment,
                    corrected_second_moment.sqrt().add_(eps),
                    value=-lr,
                )

        return loss


@dataclass
class OptimizerParameterGroups:
    nora: list[Tensor]
    nadamw_decay: list[Tensor]
    nadamw_no_decay: list[Tensor]


def partition_optimizer_parameters(model: nn.Module) -> OptimizerParameterGroups:
    """Partition trainable parameters by module role and parameter identity."""
    nora_ids: set[int] = set()
    no_decay_ids: set[int] = set()

    for module_name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for param in module.parameters(recurse=False):
                if param.requires_grad:
                    no_decay_ids.add(id(param))
            continue

        if isinstance(module, (Embedding, RMSNorm)):
            for param in module.parameters(recurse=False):
                if param.requires_grad:
                    no_decay_ids.add(id(param))
            continue

        if not isinstance(module, Linear):
            continue

        if module.bias is not None and module.bias.requires_grad:
            no_decay_ids.add(id(module.bias))
        if not module.weight.requires_grad:
            continue
        if module_name.rsplit(".", 1)[-1] == "lm_head":
            no_decay_ids.add(id(module.weight))
        elif module.weight.ndim == 2:
            nora_ids.add(id(module.weight))

    nora: list[Tensor] = []
    nadamw_decay: list[Tensor] = []
    nadamw_no_decay: list[Tensor] = []
    seen: set[int] = set()
    for param in model.parameters():
        param_id = id(param)
        if not param.requires_grad or param_id in seen:
            continue
        seen.add(param_id)
        if param_id in no_decay_ids or param.ndim <= 1:
            nadamw_no_decay.append(param)
        elif param_id in nora_ids:
            nora.append(param)
        else:
            nadamw_decay.append(param)

    trainable_ids = {id(param) for param in model.parameters() if param.requires_grad}
    grouped_ids = {id(param) for param in [*nora, *nadamw_decay, *nadamw_no_decay]}
    if grouped_ids != trainable_ids:
        missing = len(trainable_ids - grouped_ids)
        extra = len(grouped_ids - trainable_ids)
        raise RuntimeError(
            f"Optimizer parameter partition is incomplete: missing={missing}, extra={extra}"
        )

    return OptimizerParameterGroups(nora, nadamw_decay, nadamw_no_decay)


@OptimizerFactory.register("nora_nadamw")
class NoraNAdamW(Optimizer):
    """Nora for internal linear weights and NAdamW for remaining parameters."""

    optimizer_name = "nora_nadamw"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        nora_lr: float = 5e-3,
        nora_weight_decay: float = 0.0,
        nora_beta: float = 0.95,
        nora_momentum: float = 0.95,
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

        self.nora = (
            Nora(
                groups.nora,
                lr=nora_lr,
                weight_decay=nora_weight_decay,
                momentum=nora_momentum,
                beta=nora_beta,
            )
            if groups.nora
            else None
        )

        nadamw_groups = []
        if groups.nadamw_decay:
            nadamw_groups.append(
                {"params": groups.nadamw_decay, "weight_decay": weight_decay}
            )
        if groups.nadamw_no_decay:
            nadamw_groups.append(
                {"params": groups.nadamw_no_decay, "weight_decay": 0.0}
            )
        self.nadamw = NAdamW(nadamw_groups, lr=lr) if nadamw_groups else None
        self.param_groups = refresh_param_groups([self.nora, self.nadamw])

    @torch.no_grad()
    def step(self, closure=None):
        return composite_step(
            [opt for opt in (self.nora, self.nadamw) if opt is not None],
            closure,
        )

    def zero_grad(self, set_to_none: bool = True):
        composite_zero_grad(
            [opt for opt in (self.nora, self.nadamw) if opt is not None],
            set_to_none,
        )

    def state_dict(self) -> dict[str, Any]:
        return composite_state_dict({"nora": self.nora, "nadamw": self.nadamw})

    def load_state_dict(self, state_dict: dict[str, Any]):
        if "muon" in state_dict or "adamw" in state_dict:
            raise ValueError(
                "Checkpoint uses muon_adamw state; select optimizer='muon_adamw' "
                "to resume it"
            )
        if "nora" not in state_dict or "nadamw" not in state_dict:
            raise ValueError(
                "Checkpoint optimizer state is not compatible with nora_nadamw"
            )

        saved_nora = state_dict["nora"]
        saved_nadamw = state_dict["nadamw"]
        if (self.nora is None) != (saved_nora is None):
            raise ValueError("Checkpoint Nora parameter groups do not match the model")
        if (self.nadamw is None) != (saved_nadamw is None):
            raise ValueError(
                "Checkpoint NAdamW parameter groups do not match the model"
            )
        if self.nora is not None:
            self.nora.load_state_dict(saved_nora)
        if self.nadamw is not None:
            self.nadamw.load_state_dict(saved_nadamw)
        self.param_groups = refresh_param_groups([self.nora, self.nadamw])
