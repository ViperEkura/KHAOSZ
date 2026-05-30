"""Storage backends for different data formats.

Layers:
  - I/O layer:       save_* / load_* functions, read/write raw files (HDF5/bin)
                      return Dict[str, List[Tensor]] — format-specific, no state
  - Store (ABC):     central abstraction, normalizes multi-segment into
                      Dict[str, List[Tensor]] per key via _normalize(),
                      fetch() uses bisect across segments — no forced concat
  - Dataset layer:   BaseDataset owns a Store, only calls store.fetch(begin, end, key)

Key properties:
  - Multi-segment:   segments kept as-is, no forced concatenation — safe for
                      datasets larger than RAM
  - Explicit length: _length = min(total elements across keys), set at load,
                      __len__ returns O(1)
  - Zero-copy mmap:  MmapStore wraps np.memmap(mode="r"), all DataLoader
                      workers share OS page-cache pages
"""

import bisect
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

import h5py
import numpy as np
import torch
from torch import Tensor

from astrai.factory import BaseFactory


def save_h5(file_path: str, file_name: str, tensor_group: Dict[str, List[Tensor]]):
    os.makedirs(file_path, exist_ok=True)
    full_file_path = os.path.join(file_path, f"{file_name}.h5")
    with h5py.File(full_file_path, "w") as f:
        for key, tensors in tensor_group.items():
            grp = f.create_group(key)
            for idx, tensor in enumerate(tensors):
                arr = tensor.cpu().numpy()
                grp.create_dataset(f"data_{idx}", data=arr)


def load_h5(file_path: str, share_memory=True) -> Dict[str, List[Tensor]]:
    tensor_group: Dict[str, List[Tensor]] = {}

    root_path = Path(file_path)
    h5_files = list(root_path.rglob("*.h5")) + list(root_path.rglob("*.hdf5"))

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            for key in f.keys():
                grp = f[key]
                dsets = []
                for dset_name in grp.keys():
                    dset = grp[dset_name]
                    tensor = torch.from_numpy(dset[:])
                    if share_memory:
                        tensor = tensor.share_memory_()
                    dsets.append(tensor)

                if tensor_group.get(key) is None:
                    tensor_group[key] = []
                tensor_group[key].extend(dsets)

    return tensor_group


def save_bin(file_path: str, tensor_group: Dict[str, List[Tensor]]):
    os.makedirs(file_path, exist_ok=True)
    meta = {}
    for key, tensors in tensor_group.items():
        cat = torch.cat(tensors, dim=0)
        meta[key] = {"shape": list(cat.shape), "dtype": str(cat.dtype).split(".")[-1]}
        np.asarray(cat.cpu().numpy()).tofile(os.path.join(file_path, f"{key}.bin"))
    with open(os.path.join(file_path, "meta.json"), "w") as f:
        json.dump(meta, f)


def load_bin(file_path: str) -> Dict[str, List[Tensor]]:
    with open(os.path.join(file_path, "meta.json"), "r") as f:
        meta = json.load(f)
    segments: Dict[str, List[Tensor]] = {}
    for key, info in meta.items():
        arr = np.memmap(
            os.path.join(file_path, f"{key}.bin"),
            dtype=info["dtype"],
            mode="r+",
            shape=tuple(info["shape"]),
        )
        segments[key] = [torch.from_numpy(arr)]
    return segments


def detect_format(load_path: str) -> str:
    """Auto-detect storage format from files in the directory.

    Args:
        load_path: Directory or file path

    Returns:
        Format string ("h5" or "bin")

    Raises:
        FileNotFoundError: If no supported data files are found
    """
    root = Path(load_path)
    if root.is_file():
        suffix = root.suffix.lower()
        if suffix in (".h5", ".hdf5"):
            return "h5"
        raise ValueError(f"Unsupported file format: {suffix}")

    h5_files = list(root.rglob("*.h5")) + list(root.rglob("*.hdf5"))
    if h5_files:
        return "h5"
    bin_files = list(root.rglob("*.bin"))
    if bin_files:
        has_meta = (root / "meta.json").exists() or len(
            list(root.rglob("meta.json"))
        ) > 0
        if has_meta:
            return "bin"
    raise FileNotFoundError(f"No supported data files found at {load_path}")


class Store(ABC):
    """String keys -> segmented tensors with ``fetch(begin, end, keys)``.

    Each key maps to one or more tensor segments (no forced concatenation).
    ``len(store)`` returns ``self._length`` (explicit, O(1)), the minimum
    total element count across all keys.

    Subclasses fill ``self._data`` and ``self._cum`` during ``load()``
    via ``_normalize()``.
    """

    def __init__(self):
        self._data: Dict[str, List[Tensor]] = {}
        self._cum: Dict[str, List[int]] = {}
        self._length: int = 0

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @property
    def keys(self) -> List[str]:
        return list(self._data.keys())

    def __len__(self) -> int:
        return self._length

    def fetch(
        self,
        begin: int,
        end: int,
        keys: Union[str, List[str]],
    ):
        if not self._data:
            raise RuntimeError("Store not loaded")
        if not (0 <= begin < self._length and 0 <= end <= self._length):
            raise ValueError(
                f"Index out of bounds: begin={begin}, end={end}, length={self._length}"
            )
        if isinstance(keys, str):
            return self._fetch_key(keys, begin, end)
        return {k: self._fetch_key(k, begin, end) for k in keys}

    def _fetch_key(self, key: str, begin: int, end: int) -> Tensor:
        """Fetch slice [begin, end) across potentially multiple segments."""
        segments = self._data[key]
        cum = self._cum[key]
        seg_start = bisect.bisect_right(cum, begin)
        seg_end = bisect.bisect_left(cum, end)

        results = []
        for i in range(seg_start, seg_end + 1):
            prev = cum[i - 1] if i > 0 else 0
            s = max(begin - prev, 0)
            e = min(end - prev, segments[i].shape[0])
            results.append(segments[i][s:e])

        return results[0] if len(results) == 1 else torch.cat(results, dim=0)

    def _normalize(self, raw: Dict[str, List[Tensor]]):
        """Register segments and pre-compute cumulative lengths.

        Does NOT concatenate — segments are kept as-is to avoid OOM on
        large datasets.  Sets ``self._length`` to the minimum total
        element count across all keys.
        """
        for key, tensors in raw.items():
            self._data[key] = tensors
            cum = []
            total = 0
            for t in tensors:
                total += t.shape[0]
                cum.append(total)
            self._cum[key] = cum
        self._length = (
            min((cum[-1] if cum else 0) for cum in self._cum.values())
            if self._cum
            else 0
        )


class StoreFactory(BaseFactory["Store"]):
    """Factory for creating Store instances by type name.

    Example::

        @StoreFactory.register("custom")
        class CustomStore(Store):
            ...
    """

    @classmethod
    def _validate_component(cls, store_cls: type):
        if not issubclass(store_cls, Store):
            raise TypeError(f"{store_cls.__name__} must inherit from Store")


@StoreFactory.register("h5")
class H5Store(Store):
    """HDF5-based storage backend (pre-tokenized data)."""

    def load(self, path: str):
        self._normalize(load_h5(path))


@StoreFactory.register("bin")
class MmapStore(Store):
    """Memory-mapped binary storage backend.

    Each key is a single .bin file backed by ``np.memmap(mode="r")``.
    No per-process memory duplication — all DataLoader workers share the
    same OS page-cache pages.

    Format on disk::

        data_root/
          meta.json          # {key: {shape, dtype}, ...}
          <key>.bin          # raw numpy array, one per key
    """

    def load(self, path: str):
        self._mmap_refs = []
        root = Path(path)
        all_raw: Dict[str, List[Tensor]] = {}
        meta_paths = list(root.rglob("meta.json"))
        for meta_path in meta_paths:
            raw = load_bin(str(meta_path.parent))
            for key, tensors in raw.items():
                if key not in all_raw:
                    all_raw[key] = []
                all_raw[key].extend(tensors)
        if not meta_paths:
            raise FileNotFoundError(f"No meta.json found under {path}")
        self._normalize(all_raw)
        for tensors in self._data.values():
            self._mmap_refs.extend(tensors)
