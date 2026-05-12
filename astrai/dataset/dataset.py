"""Dataset implementations with factory pattern for training."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from astrai.dataset.storage import (
    BaseStorage,
    create_storage,
    detect_format,
)
from astrai.factory import BaseFactory


class BaseDataset(Dataset, ABC):
    """Abstract base class for all dataset types.

    Implements common functionality for window-based data fetching.
    Uses a storage abstraction for format-agnostic data loading.
    """

    def __init__(self, window_size: int, stride: int):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.storage: Optional[BaseStorage] = None

    def load(self, load_path: str, storage_type: Optional[str] = None, tokenizer=None):
        """Load dataset from the given path.

        Auto-detects the storage format if not specified.

        Args:
            load_path: Path to the data directory or file
            storage_type: Force a specific storage type ("h5", "json"),
                          or None for auto-detection
            tokenizer: Callable str -> List[int], used to tokenize raw text
                       in JSON files. Ignored for HDF5.
        """
        if storage_type is None:
            storage_type = detect_format(load_path)
        self.storage = create_storage(storage_type)
        self.storage.load(load_path, tokenizer=tokenizer)

    def load_json(self, load_path: str, tokenizer=None):
        """Load dataset from JSON files explicitly.

        Args:
            load_path: Path to the JSON data file or directory
            tokenizer: Optional tokenizer callable for raw text JSON.
        """
        self.load(load_path, storage_type="json", tokenizer=tokenizer)

    @property
    def count(self) -> int:
        """Return the total number of raw elements (tokens) in the dataset."""
        if self.storage is None:
            return 0
        return len(self.storage)

    @property
    def keys(self) -> List[str]:
        """Return the available data keys."""
        if self.storage is None:
            return []
        return self.storage.keys

    def get_index(self, index: int) -> tuple:
        """Calculate begin and end indices for a sample.

        Args:
            index: Sample index

        Returns:
            Tuple of (begin_idx, end_idx)
        """
        assert self.storage is not None
        total = len(self.storage)
        assert total > self.window_size

        begin_idx = min(index * self.stride, total - 1 - self.window_size)
        end_idx = min(begin_idx + self.window_size, total - 1)

        return begin_idx, end_idx

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Get a single sample by index.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        assert self.storage is not None
        total = len(self.storage)
        if total <= self.window_size:
            return 0
        return (total - 1 - self.window_size) // self.stride + 1


class DatasetFactory(BaseFactory["BaseDataset"]):
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

    @classmethod
    def _validate_component(cls, dataset_cls: type) -> None:
        """Validate that the dataset class inherits from BaseDataset."""
        if not issubclass(dataset_cls, BaseDataset):
            raise TypeError(f"{dataset_cls.__name__} must inherit from BaseDataset")

    @classmethod
    def create(cls, train_type: str, window_size: int, stride: int) -> "BaseDataset":
        """Create a dataset instance.

        Args:
            train_type: Type of training ("seq", "sft", "dpo", "grpo")
            window_size: Window size for data sampling
            stride: Stride between consecutive samples

        Returns:
            Dataset instance
        """
        return super().create(train_type, window_size, stride)

    @classmethod
    def load(
        cls,
        train_type: str,
        load_path: str,
        window_size: int,
        stride: Optional[int] = None,
        storage_type: Optional[str] = None,
        tokenizer=None,
    ) -> "BaseDataset":
        """Create and load a dataset in one step.

        Args:
            train_type: Type of training dataset
            load_path: Path to the data file
            window_size: Window size for data sampling
            stride: Stride between consecutive samples (default: same as window_size)
            storage_type: Storage type ("h5", "json") or None for auto-detection
            tokenizer: Callable str -> List[int] for raw text JSON tokenization

        Returns:
            Loaded dataset instance
        """
        if stride is None:
            stride = window_size

        dataset = cls.create(train_type, window_size, stride)
        dataset.load(load_path, storage_type=storage_type, tokenizer=tokenizer)

        return dataset

    @classmethod
    def available_types(cls) -> list:
        """Return list of registered dataset type names."""
        return cls.list_registered()


@DatasetFactory.register("seq")
class SEQDataset(BaseDataset):
    """Dataset for sequential next-token prediction training."""

    def __init__(self, window_size: int, stride: int):
        super().__init__(window_size, stride)

    def _fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        return self.storage.fetch(begin_idx, end_idx, "sequence")

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
        return self.storage.fetch(begin_idx, end_idx, key)

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
        return self.storage.fetch(begin_idx, end_idx, key)

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
        return self.storage.fetch(begin_idx, end_idx, key)

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
