"""Momentum, orthogonalization and local shard application for Muon."""

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, optim
from torch.distributed.tensor import DTensor
from torch.optim._muon import _adjust_lr, _single_tensor_muon

from .strategies.batching import MatrixTriple, batched_sharded_matrices
from .strategies.dtensor import DTensorTransferPolicy, FullTensorTransferPolicy
from .strategies.orthogonalization import (
    KernelOrthogonalizer,
    TorchOrthogonalizer,
    resolve_backend,
    single_tensor_muon_reuse_buffers,
)


def step_sharded_muon(
    chunks: Iterable[list[MatrixTriple]],
    group: Mapping,
    orthogonalize,
    transfer_policy: DTensorTransferPolicy | None = None,
) -> None:
    """Apply sharded updates with independent NS and transfer strategies."""
    transfer = (
        transfer_policy if transfer_policy is not None else FullTensorTransferPolicy()
    )
    lr = group["lr"].item() if isinstance(group["lr"], Tensor) else group["lr"]
    momentum = group["momentum"]
    nesterov = group["nesterov"]
    for chunk in chunks:
        updates = []
        for _, grad, buf in chunk:
            buf.lerp_(grad, 1 - momentum)
            updates.append(grad.lerp(buf, momentum) if nesterov else buf)
        packed = torch.stack(updates) if len(updates) > 1 else updates[0]
        full = transfer.gather(packed)
        ortho = orthogonalize(
            full, group["ns_coefficients"], group["ns_steps"], group["eps"]
        )
        # Preserve the one-rank BF16 momentum state transition after packing.
        if (
            len(updates) > 1
            and not nesterov
            and full.dtype == torch.bfloat16
            and packed.device_mesh.size() == 1
        ):
            for index, (_, _, buf) in enumerate(chunk):
                buf.to_local().copy_(full[index])
        sharded = transfer.redistribute(ortho, packed)
        results = torch.unbind(sharded, dim=0) if len(updates) > 1 else (sharded,)
        for (param, _, _), update in zip(chunk, results):
            adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
            param.mul_(1 - lr * group["weight_decay"])
            param.add_(update, alpha=-adjusted_lr)


class _ShardedMuon(optim.Muon):
    """Muon that gathers matrix DTensors for global Newton-Schulz matmuls."""

    def __init__(
        self,
        params,
        *,
        reuse_ns_buffers: bool = False,
        use_ns_kernels: bool = False,
        ns_batch_size: int = 4,
        ns_backend: str | None = None,
        batch_ns_updates: bool | None = None,
        dtensor_transfer_policy: DTensorTransferPolicy | None = None,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.reuse_ns_buffers = reuse_ns_buffers
        self.use_ns_kernels = use_ns_kernels
        if not isinstance(ns_batch_size, int) or ns_batch_size < 1:
            raise ValueError("ns_batch_size must be a positive integer")
        if batch_ns_updates is not None and not isinstance(batch_ns_updates, bool):
            raise ValueError("batch_ns_updates must be a boolean or None")
        self.ns_batch_size = ns_batch_size
        resolve_backend(use_ns_kernels, ns_backend)
        self.ns_backend = ns_backend
        self.batch_ns_updates = batch_ns_updates
        self.dtensor_transfer_policy = dtensor_transfer_policy

    def _resolved_backend(self) -> str:
        return resolve_backend(self.use_ns_kernels, self.ns_backend)

    def _should_batch(self) -> bool:
        return (
            self.batch_ns_updates
            if self.batch_ns_updates is not None
            else self.use_ns_kernels
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            bufs: list[Tensor] = []
            self._init_group(group, params, grads, bufs)
            plain, sharded = [], []
            for param, grad, buf in zip(params, grads, bufs):
                (sharded if isinstance(param, DTensor) else plain).append(
                    (param, grad, buf)
                )
            backend = self._resolved_backend()
            batch_updates = self._should_batch()
            if plain:
                pp, gg, bb = (list(t) for t in zip(*plain))
                if self.reuse_ns_buffers or backend == "auto" or batch_updates:
                    single_tensor_muon_reuse_buffers(
                        pp,
                        gg,
                        bb,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        ns_coefficients=group["ns_coefficients"],
                        ns_steps=group["ns_steps"],
                        eps=group["eps"],
                        adjust_lr_fn=group["adjust_lr_fn"],
                        has_complex=False,
                        backend=backend,
                        batch_ns_updates=batch_updates,
                        ns_batch_size=self.ns_batch_size,
                    )
                else:
                    _single_tensor_muon(
                        pp,
                        gg,
                        bb,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        ns_coefficients=group["ns_coefficients"],
                        ns_steps=group["ns_steps"],
                        eps=group["eps"],
                        adjust_lr_fn=group["adjust_lr_fn"],
                        has_complex=False,
                    )
            if sharded:
                chunks = batched_sharded_matrices(
                    sharded, self.ns_batch_size if batch_updates else 1
                )
                orthogonalize = (
                    KernelOrthogonalizer()
                    if backend == "auto"
                    else TorchOrthogonalizer()
                )
                step_sharded_muon(
                    chunks, group, orthogonalize, self.dtensor_transfer_policy
                )
        return loss


_step_sharded_muon = step_sharded_muon
_batched_sharded_matrices = batched_sharded_matrices
