"""Dataset implementations with factory pattern for training."""

import torch
import bisect

from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import Dataset
from khaosz.data.serialization import load_h5
from typing import List, Dict, Optional, Union


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
        """Fetch data in the range [begin_idx, end_idx).

        Args:
            begin_idx: Starting index (inclusive)
            end_idx: Ending index (exclusive)

        Returns:
            Concatenated tensor of data in the specified range
        """
        if not (
            0 <= begin_idx < self.total_length and 0 <= end_idx <= self.total_length
        ):
            raise ValueError("begin_idx or end_idx out of bounds")
        if begin_idx >= end_idx:
            return torch.tensor([], dtype=torch.long)

        # Find segment boundaries for the range
        seg_start_idx = bisect.bisect_right(self.cum_lengths, begin_idx)
        seg_end_idx = bisect.bisect_left(self.cum_lengths, end_idx)

        result_segments = []

        for i in range(seg_start_idx, seg_end_idx + 1):
            prev_cum = self.cum_lengths[i - 1] if i > 0 else 0
            start = max(begin_idx - prev_cum, 0)
            end = min(end_idx - prev_cum, len(self.segments[i]))
            data = self.segments[i][start:end]
            result_segments.append(data)

        return torch.cat(result_segments, dim=0)


class MultiSegmentFetcher:
    """Manages multiple segment fetchers for different data keys.

    Each key corresponds to a different type of data (e.g., "sequence", "mask").
    """

    def __init__(self, muti_segments: Dict):
        self.muti_keys = list(muti_segments.keys())
        self.muti_fetchers = {
            key: BaseSegmentFetcher(segments) for key, segments in muti_segments.items()
        }

    def __len__(self) -> int:
        """Returns the minimum length across all fetchers."""
        len_list = [len(seg) for seg in self.muti_fetchers.values()]
        return min(len_list)

    def key_fetch(
        self, begin_idx: int, end_idx: int, keys: Union[str, List[str]]
    ) -> Dict:
        """Fetch data for specific keys.

        Args:
            begin_idx: Starting index
            end_idx: Ending index
            keys: Single key or list of keys to fetch

        Returns:
            Dictionary of tensors if multiple keys, single tensor if one key
        """
        fetch_dict = {}
        keys = [keys] if isinstance(keys, str) else keys

        for key in keys:
            fetcher = self.muti_fetchers[key]
            fetch_tensor = fetcher.fetch_data(begin_idx, end_idx)
            fetch_dict[key] = fetch_tensor

        return fetch_dict if len(keys) > 1 else fetch_dict[keys[0]]

    def fetch_data(self, begin_idx: int, end_idx: int) -> Dict:
        """Fetch all keys."""
        return self.key_fetch(begin_idx, end_idx, self.muti_keys)


class BaseDataset(Dataset, ABC):
    """Abstract base class for all dataset types.

    Implements common functionality for window-based data fetching.
    """

    def __init__(self, window_size: int, stride: int):
        super().__init__()
        self.segments = {}
        self.window_size = window_size
        self.stride = stride
        self.total_samples = None
        self.fetcher: Optional[MultiSegmentFetcher] = None

    def load(self, load_path: str):
        """Load dataset from HDF5 file.

        Args:
            load_path: Path to the HDF5 data file
        """
        self.segments = load_h5(load_path)
        self.fetcher = MultiSegmentFetcher(self.segments)
        self.total_samples = len(self.fetcher)

    def get_index(self, index: int) -> tuple:
        """Calculate begin and end indices for a sample.

        Args:
            index: Sample index

        Returns:
            Tuple of (begin_idx, end_idx)
        """
        assert self.total_samples > self.window_size

        begin_idx = min(index * self.stride, self.total_samples - 1 - self.window_size)
        end_idx = min(begin_idx + self.window_size, self.total_samples - 1)

        return begin_idx, end_idx

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Get a single sample by index.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        assert self.total_samples is not None
        if self.total_samples <= self.window_size:
            return 0
        return (self.total_samples - 1 - self.window_size) // self.stride + 1


class DatasetFactory:
    """Factory class for creating dataset instances.

    Supports decorator-based registration for extensible dataset types.
    All default dataset types (seq, sft, dpo, grpo) are registered automatically
    when their classes are defined with the decorator.

    Example usage:
        @DatasetFactory.register("custom")
        class CustomDataset(BaseDataset):
            ...

        dataset = DatasetFactory.create("custom", window_size, stride)
    """

    SUPPORTED_TYPES = frozenset({"seq", "sft", "dpo", "grpo"})
    DATASET_MAP: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a new dataset class.

        Args:
            name: Registration name for the dataset type

        Returns:
            Decorator function that registers the dataset class
        """

        def decorator(dataset_cls: type) -> type:
            if not issubclass(dataset_cls, BaseDataset):
                raise TypeError(f"{dataset_cls.__name__} must inherit from BaseDataset")
            cls.DATASET_MAP[name] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def create(cls, train_type: str, window_size: int, stride: int) -> BaseDataset:
        """Create a dataset instance.

        Args:
            train_type: Type of training ("seq", "sft", "dpo", "grpo")
            window_size: Window size for data sampling
            stride: Stride between consecutive samples

        Returns:
            Dataset instance
        """
        if train_type not in cls.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown dataset type: '{train_type}'. "
                f"Supported types: {sorted(cls.SUPPORTED_TYPES)}"
            )

        if train_type not in cls.DATASET_MAP:
            raise NotImplementedError(
                f"Dataset type '{train_type}' is supported but not yet implemented."
            )

        dataset_cls = cls.DATASET_MAP[train_type]
        return dataset_cls(window_size, stride)

    @classmethod
    def load(
        cls,
        train_type: str,
        load_path: str,
        window_size: int,
        stride: Optional[int] = None,
    ) -> BaseDataset:
        """Create and load a dataset in one step.

        Args:
            train_type: Type of training dataset
            load_path: Path to the data file
            window_size: Window size for data sampling
            stride: Stride between consecutive samples (default: same as window_size)

        Returns:
            Loaded dataset instance
        """
        if stride is None:
            stride = window_size

        dataset = cls.create(train_type, window_size, stride)
        dataset.load(load_path)

        return dataset

    @classmethod
    def available_types(cls) -> list:
        """Return list of registered dataset type names."""
        return list(cls.DATASET_MAP.keys())


# ============== Dataset Classes ==============
# All dataset classes are registered at class definition time using the decorator


@DatasetFactory.register("seq")
class SEQDataset(BaseDataset):
    """Dataset for sequential next-token prediction training."""

    def __init__(self, window_size: int, stride: int):
        super().__init__(window_size, stride)

    def _fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, "sequence")

    def __getitem__(self, index):
        begin_idx, end_idx = self.get_index(index)

        x = self._fetch_data(begin_idx, end_idx).to(dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1).to(dtype=torch.long)

        return {"input_ids": x, "target_ids": y}


@DatasetFactory.register("sft")
class SFTDataset(BaseDataset):
    """Dataset for supervised fine-tuning with loss masking."""

    def __init__(self, window_size: int, stride: int):
        super().__init__(window_size, stride)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)

    def __getitem__(self, index):
        begin_idx, end_idx = self.get_index(index)

        x = self._fetch_data(begin_idx, end_idx, "sequence").to(dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1, "sequence").to(
            dtype=torch.long
        )
        loss_mask = self._fetch_data(begin_idx + 1, end_idx + 1, "loss_mask").to(
            dtype=torch.bool
        )

        return {"input_ids": x, "target_ids": y, "loss_mask": loss_mask}


@DatasetFactory.register("dpo")
class DPODataset(BaseDataset):
    """Dataset for Direct Preference Optimization training."""

    def __init__(self, window_size: int, stride: int):
        super().__init__(window_size, stride)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)

    def __getitem__(self, index: int):
        begin_idx, end_idx = self.get_index(index)

        chosen = self._fetch_data(begin_idx, end_idx, "chosen").to(dtype=torch.long)
        rejected = self._fetch_data(begin_idx, end_idx, "rejected").to(dtype=torch.long)
        chosen_mask = self._fetch_data(begin_idx, end_idx, "chosen_mask").to(
            dtype=torch.bool
        )
        rejected_mask = self._fetch_data(begin_idx, end_idx, "rejected_mask").to(
            dtype=torch.bool
        )

        return {
            "chosen": chosen,
            "rejected": rejected,
            "chosen_mask": chosen_mask,
            "rejected_mask": rejected_mask,
        }


@DatasetFactory.register("grpo")
class GRPODataset(BaseDataset):
    """Dataset for Group Relative Policy Optimization training."""

    def __init__(self, window_size: int, stride: int):
        super().__init__(window_size, stride)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        begin_idx, end_idx = self.get_index(index)

        prompts = self._fetch_data(begin_idx, end_idx, "prompts")
        responses = self._fetch_data(begin_idx, end_idx, "responses")
        masks = self._fetch_data(begin_idx, end_idx, "masks")
        rewards = self._fetch_data(begin_idx, end_idx, "rewards")

        return {
            "prompts": prompts,
            "responses": responses,
            "masks": masks,
            "rewards": rewards,
        }


# Backward compatibility alias
DatasetLoader = DatasetFactory
