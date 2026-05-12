"""Storage backends for different data formats.

Each storage handles format-specific loading (HDF5, JSON, etc.) and provides
a uniform interface for data access and length observation via fetchers.
"""

import bisect
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import h5py
import torch
from torch import Tensor


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


def save_json(file_path: str, file_name: str, tensor_group: Dict[str, List[Tensor]]):
    os.makedirs(file_path, exist_ok=True)
    full_file_path = os.path.join(file_path, f"{file_name}.json")
    json_data = {}
    for key, tensors in tensor_group.items():
        json_data[key] = [tensor.tolist() for tensor in tensors]
    with open(full_file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False)


def load_json(
    file_path: str,
    share_memory: bool = True,
    tokenizer: Optional[Callable[[str], List[int]]] = None,
) -> Dict[str, List[Tensor]]:
    """Load tensor data from JSON files.

    Supports two modes:
    - Pre-tokenized: JSON values are List[List[int]] (token IDs), loaded as-is.
    - Raw text: JSON values are List[str], tokenized via ``tokenizer`` callable
      at load time. A ``tokenizer`` receives a str and returns List[int].

    Non-data JSON files (e.g. config.json) with scalar/object values are
    silently skipped.
    """
    tensor_group: Dict[str, List[Tensor]] = {}
    root_path = Path(file_path)
    json_files = list(root_path.rglob("*.json")) + list(root_path.rglob("*.jsonl"))
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        for key, sequences in data.items():
            if not isinstance(sequences, list):
                continue
            tensors = []
            for seq in sequences:
                if tokenizer is not None and isinstance(seq, str):
                    seq = tokenizer(seq)
                tensor = torch.tensor(seq, dtype=torch.long)
                if share_memory:
                    tensor = tensor.share_memory_()
                tensors.append(tensor)
            if tensor_group.get(key) is None:
                tensor_group[key] = []
            tensor_group[key].extend(tensors)
    return tensor_group


def detect_format(load_path: str) -> str:
    """Auto-detect storage format from files in the directory.

    Args:
        load_path: Directory or file path

    Returns:
        Format string ("h5" or "json")

    Raises:
        FileNotFoundError: If no supported data files are found
    """
    root = Path(load_path)
    if root.is_file():
        suffix = root.suffix.lower()
        if suffix in (".h5", ".hdf5"):
            return "h5"
        if suffix in (".json", ".jsonl"):
            return "json"
        raise ValueError(f"Unsupported file format: {suffix}")

    h5_files = list(root.rglob("*.h5")) + list(root.rglob("*.hdf5"))
    if h5_files:
        return "h5"
    json_files = list(root.rglob("*.json")) + list(root.rglob("*.jsonl"))
    if json_files:
        return "json"
    raise FileNotFoundError(f"No supported data files found at {load_path}")


class BaseSegmentFetcher:
    """Fetches data segments across multiple tensor segments.

    Maintains cumulative lengths for efficient range queries across
    multiple discontinuous segments.
    """

    def __init__(self, segments: List[Tensor]):
        self.segments = segments
        self.cum_lengths = []

        total = 0
        for seg in segments:
            total += torch.numel(seg)
            self.cum_lengths.append(total)

        self.total_length = total

    def __len__(self) -> int:
        return self.total_length

    def fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        """Fetch data in the range [begin_idx, end_idx)."""
        if not (
            0 <= begin_idx < self.total_length and 0 <= end_idx <= self.total_length
        ):
            raise ValueError("begin_idx or end_idx out of bounds")
        if begin_idx >= end_idx:
            return torch.tensor([], dtype=torch.long)

        seg_start_idx = bisect.bisect_right(self.cum_lengths, begin_idx)
        seg_end_idx = bisect.bisect_left(self.cum_lengths, end_idx)

        result_segments = []

        for i in range(seg_start_idx, seg_end_idx + 1):
            prev_cum = self.cum_lengths[i - 1] if i > 0 else 0
            start = max(begin_idx - prev_cum, 0)
            end = min(end_idx - prev_cum, len(self.segments[i]))
            result_segments.append(self.segments[i][start:end])

        return torch.cat(result_segments, dim=0)


class MultiSegmentFetcher:
    """Manages multiple segment fetchers for different data keys."""

    def __init__(self, multi_segments: Dict):
        self.multi_keys = list(multi_segments.keys())
        self.multi_fetchers = {
            key: BaseSegmentFetcher(segments)
            for key, segments in multi_segments.items()
        }

    def __len__(self) -> int:
        """Returns the minimum length across all fetchers."""
        if not self.multi_fetchers:
            return 0
        len_list = [len(seg) for seg in self.multi_fetchers.values()]
        return min(len_list)

    def key_fetch(
        self, begin_idx: int, end_idx: int, keys: Union[str, List[str]]
    ) -> Dict:
        """Fetch data for specific keys."""
        fetch_dict = {}
        keys = [keys] if isinstance(keys, str) else keys

        for key in keys:
            fetcher = self.multi_fetchers[key]
            fetch_tensor = fetcher.fetch_data(begin_idx, end_idx)
            fetch_dict[key] = fetch_tensor

        return fetch_dict if len(keys) > 1 else fetch_dict[keys[0]]

    def fetch_data(self, begin_idx: int, end_idx: int) -> Dict:
        """Fetch all keys."""
        return self.key_fetch(begin_idx, end_idx, self.multi_keys)


class BaseStorage(ABC):
    """Abstract storage backend for loading and dispatching data.

    Storage encapsulates format-specific loading and provides a uniform
    interface for data access and length observation. Subclasses handle
    different data formats (HDF5, JSON, etc.) while exposing the same
    fetch interface.
    """

    def __init__(self):
        self._fetcher: Optional[MultiSegmentFetcher] = None

    @abstractmethod
    def load(self, load_path: str, tokenizer=None) -> None:
        """Load data from the given path into internal fetcher."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Total number of raw elements (tokens) in storage."""
        if self._fetcher is None:
            return 0
        return len(self._fetcher)

    def fetch(self, begin_idx: int, end_idx: int, keys: Union[str, List[str]]):
        """Fetch data for the given keys and index range.

        Args:
            begin_idx: Starting index (inclusive)
            end_idx: Ending index (exclusive)
            keys: Single key or list of keys to fetch

        Returns:
            Tensor if single key, Dict[str, Tensor] if multiple keys
        """
        if self._fetcher is None:
            raise RuntimeError("Storage not loaded")
        return self._fetcher.key_fetch(begin_idx, end_idx, keys)

    @property
    def keys(self) -> List[str]:
        """Return the data keys available in this storage."""
        if self._fetcher is None:
            return []
        return self._fetcher.multi_keys


class H5Storage(BaseStorage):
    """HDF5-based storage backend (pre-tokenized data)."""

    def load(self, load_path: str, tokenizer=None) -> None:
        segments = load_h5(load_path)
        self._fetcher = MultiSegmentFetcher(segments)


class JSONStorage(BaseStorage):
    """JSON-based storage backend.

    Supports two modes:
    - Pre-tokenized: JSON values are List[List[int]], loaded as-is.
    - Raw text: JSON values are List[str], tokenized via ``tokenizer``
      callable (str -> List[int]) at load time.
    """

    def load(self, load_path: str, tokenizer=None) -> None:
        segments = load_json(load_path, tokenizer=tokenizer)
        self._fetcher = MultiSegmentFetcher(segments)


_STORAGE_REGISTRY: Dict[str, type] = {
    "h5": H5Storage,
    "json": JSONStorage,
}


def create_storage(storage_type: str) -> BaseStorage:
    """Create a storage instance by type name.

    Args:
        storage_type: Storage type name ("h5", "json")

    Returns:
        Storage instance

    Raises:
        ValueError: If the storage type is unknown
    """
    storage_cls = _STORAGE_REGISTRY.get(storage_type)
    if storage_cls is None:
        raise ValueError(
            f"Unknown storage type: '{storage_type}'. "
            f"Available: {sorted(_STORAGE_REGISTRY.keys())}"
        )
    return storage_cls()


def available_storage_types() -> List[str]:
    """Return list of registered storage type names."""
    return sorted(_STORAGE_REGISTRY.keys())
