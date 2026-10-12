"""Independent Torch and extension Newton-Schulz strategies."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

from astrai.extension.backend.newton_schulz import newton_schulz

Orthogonalizer = Callable[[Tensor, tuple[float, float, float], int, float], Tensor]


class TorchOrthogonalizer:
    """Use PyTorch Muon's Newton-Schulz recurrence."""

    def __call__(
        self,
        matrix: Tensor,
        coefficients: tuple[float, float, float],
        steps: int,
        eps: float,
    ) -> Tensor:
        if matrix.ndim == 2:
            return _zeropower_via_newtonschulz(matrix, coefficients, steps, eps)
        return torch.stack(
            [
                _zeropower_via_newtonschulz(item, coefficients, steps, eps)
                for item in matrix.unbind(0)
            ]
        )


class KernelOrthogonalizer:
    """Use AstrAI's backend dispatcher for Newton-Schulz."""

    def __call__(
        self,
        matrix: Tensor,
        coefficients: tuple[float, float, float],
        steps: int,
        eps: float,
    ) -> Tensor:
        return newton_schulz(matrix, coefficients, steps, eps, backend="auto")


def resolve_backend(use_ns_kernels: bool, ns_backend: str | None) -> str:
    """Resolve explicit backend selection with the legacy boolean fallback."""
    if ns_backend is None:
        return "auto" if use_ns_kernels else "torch"
    if ns_backend == "cuda":
        return "auto"
    if ns_backend not in ("torch", "auto"):
        raise ValueError("ns_backend must be torch, cuda or auto")
    return ns_backend


def _scalar_lr(lr: Any) -> float:
    return lr.item() if isinstance(lr, Tensor) else lr


def single_tensor_muon_reuse_buffers(
    params: list[Tensor],
    grads: list[Tensor],
    bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
    backend: str = "torch",
    batch_ns_updates: bool = False,
    ns_batch_size: int = 4,
) -> None:
    """Apply Muon updates with reusable NS buffers and optional matrix batches."""
    if has_complex:
        raise ValueError("Complex parameters are not supported")
    lr = _scalar_lr(lr)
    buckets: dict[tuple[Any, ...], list[tuple[Tensor, Tensor, Tensor]]] = {}
    for param, grad, buf in zip(params, grads, bufs):
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")
        key = (tuple(grad.shape), grad.device, grad.dtype, buf.dtype, param.dtype)
        buckets.setdefault(key, []).append((param, grad, buf))
    batch_size = ns_batch_size if batch_ns_updates else 1
    for bucket in buckets.values():
        for start in range(0, len(bucket), batch_size):
            chunk = bucket[start : start + batch_size]
            updates = []
            for param, grad, buf in chunk:
                buf.lerp_(grad, 1 - momentum)
                updates.append(grad.lerp(buf, momentum) if nesterov else buf)
            packed = torch.stack(updates) if len(chunk) > 1 else updates[0]
            output = newton_schulz(
                packed, ns_coefficients, ns_steps, eps, backend=backend
            )
            for index, (param, grad, buf) in enumerate(chunk):
                update = output[index] if len(chunk) > 1 else output
                # Preserve BF16 normalization of non-Nesterov momentum storage.
                if len(chunk) > 1 and not nesterov and buf.dtype == torch.bfloat16:
                    buf.copy_(packed[index])
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
                param.mul_(1 - lr * weight_decay)
                param.add_(update, alpha=-adjusted_lr)


_single_tensor_muon_reuse_buffers = single_tensor_muon_reuse_buffers
