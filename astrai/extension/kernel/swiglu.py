"""Fused CUDA SwiGLU with a registered first-order backward."""

from typing import Tuple

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from astrai.extension.runtime.loader import get_module


@torch.library.custom_op("astrai::swiglu", mutates_args=())
def swiglu(up: Tensor, gate: Tensor) -> Tensor:
    """Run one native CUDA kernel for the SwiGLU forward."""
    return get_module("swiglu").swiglu(up, gate)


@swiglu.register_fake
def _swiglu_fake(up: Tensor, gate: Tensor) -> Tensor:
    return torch.empty_like(up)


def _setup_context(ctx, inputs, output) -> None:
    up, gate = inputs
    ctx.save_for_backward(up, gate)


@once_differentiable
def _backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
    up, gate = ctx.saved_tensors
    return tuple(
        get_module("swiglu").swiglu_backward(grad_output.contiguous(), up, gate)
    )


swiglu.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["swiglu"]
