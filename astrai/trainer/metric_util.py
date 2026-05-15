from typing import Any, Callable, Dict

import torch
import torch.nn as nn


def _grad_stat(
    model: nn.Module, fn: Callable[[torch.Tensor], Any], default: Any
) -> dict:
    results = {}
    for name, param in model.named_parameters():
        results[name] = default
        if param.grad is not None:
            results[name] = fn(param.grad.data)
    return results


def grad_norm(model: nn.Module, norm_type: int = 2) -> Dict[str, float]:
    return _grad_stat(model, lambda g: g.norm(norm_type).item(), 0.0)


def grad_std(model: nn.Module) -> Dict[str, float]:
    return _grad_stat(model, lambda g: g.std().item(), 0.0)


def grad_max(model: nn.Module) -> Dict[str, float]:
    return _grad_stat(model, lambda g: g.max().item(), -float("inf"))


def grad_min(model: nn.Module) -> Dict[str, float]:
    return _grad_stat(model, lambda g: g.min().item(), float("inf"))


def grad_mean(model: nn.Module) -> Dict[str, float]:
    return _grad_stat(model, lambda g: g.mean().item(), 0.0)


def grad_nan_num(model: nn.Module) -> Dict[str, int]:
    return _grad_stat(model, lambda g: g.isnan().sum().item(), 0)


def ctx_get_loss(ctx):
    return ctx.loss


def ctx_get_lr(ctx):
    return ctx.optimizer.param_groups[-1]["lr"]


def ctx_get_grad_norm(ctx):
    return grad_norm(ctx.model)


def ctx_get_grad_std(ctx):
    return grad_std(ctx.model)


def ctx_get_grad_max(ctx):
    return grad_max(ctx.model)


def ctx_get_grad_min(ctx):
    return grad_min(ctx.model)


def ctx_get_grad_mean(ctx):
    return grad_mean(ctx.model)


def ctx_get_grad_nan_num(ctx):
    return grad_nan_num(ctx.model)
