"""Independent SwiGLU dispatch for training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.kernel.swiglu import swiglu as fused_swiglu
from astrai.extension.runtime.loader import is_available
from astrai.extension.runtime.operation import (
    ExplicitSelectionError,
    OpDispatcher,
    OpSpec,
)


def _torch_swiglu(up: Tensor, gate: Tensor) -> Tensor:
    return up * F.silu(gate)


@dataclass(frozen=True)
class SwiGLUSpec(OpSpec):
    hidden_dim: int
    layout: str

    @property
    def operation(self) -> str:
        return "swiglu"


def _cuda_supports(spec: SwiGLUSpec, up: Tensor, gate: Tensor) -> bool:
    return (
        spec.device.type == "cuda"
        and spec.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and up.shape == gate.shape
        and up.device == gate.device
        and up.dtype == gate.dtype
        and spec.layout == "contiguous"
        and is_available("swiglu")
    )


class SwiGLUDispatcher(OpDispatcher[SwiGLUSpec]):
    @property
    def operation(self) -> str:
        return "swiglu"

    def build_spec(self, up: Tensor, gate: Tensor) -> SwiGLUSpec:
        return SwiGLUSpec(
            device=up.device,
            dtype=up.dtype,
            phase="train" if torch.is_grad_enabled() else "infer",
            needs_backward=torch.is_grad_enabled()
            and bool(up.requires_grad or gate.requires_grad),
            hidden_dim=up.shape[-1],
            layout="contiguous"
            if up.is_contiguous() and gate.is_contiguous()
            else "strided",
        )

    def select(
        self,
        spec: SwiGLUSpec,
        choice: Any,
        origin: str,
        up: Tensor,
        gate: Tensor,
    ):
        if choice == "torch" or choice is _torch_swiglu:
            return _torch_swiglu
        if choice is None or choice == "cuda" or choice is fused_swiglu:
            if _cuda_supports(spec, up, gate):
                return fused_swiglu
            if choice is None:
                return _torch_swiglu
        elif callable(choice):
            supports = getattr(choice, "supports_call", None)
            if supports is None or supports(up, gate):
                return choice
        if origin in ("explicit", "context"):
            raise ExplicitSelectionError(
                f"Explicitly-set swiglu implementation {choice!r} "
                "cannot handle this call"
            )
        return _torch_swiglu


_swiglu_dispatcher = SwiGLUDispatcher()


def swiglu(up: Tensor, gate: Tensor) -> Tensor:
    """Apply SiLU gating through the selected forward and autograd path."""
    return _swiglu_dispatcher.dispatch(up, gate)(up, gate)


__all__ = ["SwiGLUDispatcher", "SwiGLUSpec", "swiglu"]
