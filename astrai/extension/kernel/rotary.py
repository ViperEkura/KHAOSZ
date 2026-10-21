"""Fused RoPE with a first-order gradient for the rotated tensor."""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from astrai.extension.runtime.loader import get_module


@torch.library.custom_op("astrai::rotary_emb", mutates_args=())
def rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    if freqs_cis.requires_grad:
        raise RuntimeError("CUDA rotary does not support freqs_cis gradients")
    return get_module("rotary_emb").rotary_emb(x, freqs_cis.contiguous())


@rotary_emb.register_fake
def _rotary_fake(x: Tensor, freqs_cis: Tensor) -> Tensor:
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


def _setup_context(ctx, inputs, output) -> None:
    _, freqs_cis = inputs
    ctx.save_for_backward(freqs_cis)


@once_differentiable
def _backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
    (freqs_cis,) = ctx.saved_tensors
    grad_x = get_module("rotary_emb").rotary_emb_backward(
        grad_output, freqs_cis.contiguous()
    )
    return grad_x, None


rotary_emb.register_autograd(_backward, setup_context=_setup_context)


__all__ = ["rotary_emb"]
