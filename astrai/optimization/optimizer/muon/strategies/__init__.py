"""Muon update strategies."""

from .batching import MatrixTriple, batched_sharded_matrices
from .dtensor import DTensorTransferPolicy, FullTensorTransferPolicy
from .orthogonalization import (
    KernelOrthogonalizer,
    TorchOrthogonalizer,
    resolve_backend,
    single_tensor_muon_reuse_buffers,
)

__all__ = [
    "DTensorTransferPolicy",
    "FullTensorTransferPolicy",
    "KernelOrthogonalizer",
    "MatrixTriple",
    "TorchOrthogonalizer",
    "batched_sharded_matrices",
    "resolve_backend",
    "single_tensor_muon_reuse_buffers",
]
