"""Batch compatible sharded matrices before gather and orthogonalization."""

from collections.abc import Iterator
from typing import Any

from torch import Tensor

MatrixTriple = tuple[Tensor, Tensor, Tensor]


def batched_sharded_matrices(
    triples: list[MatrixTriple], batch_size: int
) -> Iterator[list[MatrixTriple]]:
    """Group compatible matrices for shared gather and NS dispatch."""
    buckets: dict[tuple[Any, ...], list[MatrixTriple]] = {}
    for param, grad, buf in triples:
        key = (
            tuple(grad.shape),
            grad.device,
            grad.dtype,
            buf.dtype,
            param.dtype,
            id(grad.device_mesh),
            tuple(grad.placements),
        )
        buckets.setdefault(key, []).append((param, grad, buf))
    for bucket in buckets.values():
        for start in range(0, len(bucket), batch_size):
            yield bucket[start : start + batch_size]
