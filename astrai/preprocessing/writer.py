"""Storage writer strategies for pipeline output.

The :class:`StoreWriter` abstraction decouples the pipeline from the
concrete storage format (bin).  The pipeline builds a ``{key:
List[Tensor]}`` dict and delegates the write to the writer selected
by ``output.storage_format``.
"""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

from astrai.factory import BaseFactory
from astrai.serialization import save_bin

logger = logging.getLogger(__name__)


class StoreWriter(ABC):
    """Write pre-tokenized tensors to disk in a format-specific way."""

    @abstractmethod
    def save(
        self,
        output_dir: str,
        domain: str,
        shard_idx: int,
        tensors: Dict[str, List[torch.Tensor]],
        record_keys: Optional[List[str]] = None,
    ) -> None: ...


class StoreWriterFactory(BaseFactory["StoreWriter"]):
    pass


def _resolve_record_keys(
    tensors: Dict[str, List[torch.Tensor]], record_keys: Optional[List[str]]
) -> List[str]:
    keys = list(record_keys or [])
    if not keys:
        return []
    missing = set(keys).difference(tensors)
    if missing:
        raise ValueError(f"Record output is missing fields: {sorted(missing)}")
    count = len(tensors[keys[0]])
    for key in keys:
        if not count or len(tensors[key]) != count:
            raise ValueError(f"Record field '{key}' has an inconsistent record count")
        if any(
            not isinstance(tensor, torch.Tensor) or tensor.ndim != 1
            for tensor in tensors[key]
        ):
            raise ValueError(
                f"Record field '{key}' must contain one-dimensional records"
            )
    return keys


@StoreWriterFactory.register("bin")
class BinWriter(StoreWriter):
    def save(self, output_dir, domain, shard_idx, tensors, record_keys=None):
        shard_path = os.path.join(output_dir, domain, f"shard_{shard_idx:04d}")
        try:
            record_keys = _resolve_record_keys(tensors, record_keys)
            save_bin(shard_path, tensors, record_keys=record_keys)
        except Exception:
            if os.path.exists(shard_path):
                shutil.rmtree(shard_path, ignore_errors=True)
            logger.error(
                "Failed to write shard %s/%s_%04d, cleaned up partial output",
                domain,
                "shard",
                shard_idx,
                exc_info=True,
            )
            raise
