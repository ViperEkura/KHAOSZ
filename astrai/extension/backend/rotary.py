"""Rotary embedding dispatch (family "rotary").

The fused CUDA kernel handles BF16 forward and backward with gradients for x.
RotaryDispatcher owns this operation's capability checks.

Layout: x is [batch, seq_len, n_heads, head_dim] (bf16).
freqs_cis is [batch, seq_len, dim/2, 2] (f32) — [cos, sin] pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from astrai.extension.kernel.rotary import rotary_emb as _cuda_rotary
from astrai.extension.runtime.loader import is_available
from astrai.extension.runtime.operation import (
    ExplicitSelectionError,
    OpDispatcher,
    OpSpec,
)


def _torch_apply(x: Tensor, freqs_cis: Tensor) -> Tensor:
    cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]
    dtype = x.dtype
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_cis_complex = torch.complex(cos, sin).unsqueeze(-2)
    x_rotated = x_complex * freqs_cis_complex
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.to(dtype)


@dataclass(frozen=True)
class RotarySpec(OpSpec):
    head_dim: int
    layout: str

    @property
    def operation(self) -> str:
        return "rotary"


class RotaryDispatcher(OpDispatcher[RotarySpec]):
    @property
    def operation(self) -> str:
        return "rotary"

    def build_spec(self, x: Tensor, freqs_cis: Tensor) -> RotarySpec:
        return RotarySpec(
            device=x.device,
            dtype=x.dtype,
            phase="train" if torch.is_grad_enabled() else "infer",
            needs_backward=torch.is_grad_enabled()
            and bool(x.requires_grad or freqs_cis.requires_grad),
            head_dim=x.shape[-1],
            layout="contiguous" if x.is_contiguous() else "strided",
        )

    def select(
        self,
        spec: RotarySpec,
        choice: Any,
        origin: str,
        x: Tensor,
        freqs_cis: Tensor,
    ):
        cuda_capable = (
            spec.device.type == "cuda"
            and spec.dtype == torch.bfloat16
            and x.dim() in (3, 4)
            and spec.head_dim > 0
            and spec.head_dim % 4 == 0
            and x.stride(-1) == 1
            and freqs_cis.device == x.device
            and freqs_cis.dtype == torch.float32
            and not freqs_cis.requires_grad
            and freqs_cis.shape == (*x.shape[:-2], spec.head_dim // 2, 2)
            and is_available("rotary_emb")
        )
        if choice is not None:
            if choice == "torch" or choice is _torch_apply:
                return _torch_apply
            if choice == "cuda" or choice is _cuda_rotary:
                if cuda_capable:
                    return _cuda_rotary
                if origin in ("explicit", "context"):
                    raise ExplicitSelectionError(
                        "Explicitly-set rotary cuda cannot handle this call"
                    )
            elif not isinstance(choice, str):
                supports = getattr(choice, "supports_call", None)
                if supports is None or supports(x, freqs_cis):
                    return choice
                raise ExplicitSelectionError(
                    "Explicitly-set rotary implementation cannot handle this call"
                )
            elif origin in ("explicit", "context"):
                raise ValueError(f"Unknown rotary implementation: {choice!r}")
        return _cuda_rotary if cuda_capable else _torch_apply


_rotary_dispatcher = RotaryDispatcher()


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply rotary embedding to x.

    Args:
        x: [batch, seq_len, n_heads, head_dim] (bf16)
        freqs_cis: [batch, seq_len, dim/2, 2] (f32) — [cos, sin] pairs

    Returns:
        [batch, seq_len, n_heads, head_dim] (bf16)
    """
    return _rotary_dispatcher.dispatch(x, freqs_cis)(x, freqs_cis)


__all__ = ["apply_rotary_emb"]
