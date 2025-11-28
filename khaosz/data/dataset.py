import os
import json
import torch
import bisect

from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import Dataset
from typing import Callable, List, Dict, Literal, Optional, Tuple, Union

Seg = List[Tensor]  
MultiSeg = Dict[str, Seg]


def load_mmap_files(root_path: str, shared: bool=True) -> Tuple[MultiSeg, int]:
    """Load memory-mapped binary files as torch tensors.
    
    Loads configuration from file_mapper.json in the specified directory, then loads
    corresponding binary files as memory-mapped tensors. Returns tensors grouped by key
    and total number of elements.
    
    json metadata like this:
    
    ```
    [
        {
            "file_name": "file1.bin",
            "size": 1000,
            "dtype": "float32",
            "key": "key1"
        },
        ...
    ]
    ```
    
    Args:
        root_path: Root directory path containing file_mapper.json and binary files
        shared: Whether to load tensors in shared mode. If True, tensors can be 
                shared between processes
        
    Raises:
        FileNotFoundError: If file_mapper.json or any binary file in config is missing
        KeyError: If dtype in config is not in supported DTYPE_MAP
        json.JSONDecodeError: If config file is not valid JSON
        
    Returns:
        Tuple containing:
        - MultiSeg: Dictionary of tensors grouped by key, structure: {key: [tensor1, tensor2, ...]}
        - int: Total number of elements across all tensors
    """
    
    DTYPE_MAP = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    
    metadata_list = []
    mmap_shared_group: MultiSeg = {}
    
    file_mapper_path = os.path.join(root_path, "file_mapper.json")
    if not os.path.exists(file_mapper_path):
        raise FileNotFoundError(f"File mapper not found: {file_mapper_path}")
    
    with open(file_mapper_path, "r") as f:
        metadata_list = json.load(f)
    
    num_samples = sum(metadata["size"] for metadata in metadata_list)
    
    for metadata in metadata_list:
        file_path = os.path.join(root_path, metadata["file_name"])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Binary data file not found: {file_path}")
        
        size = metadata["size"]
        dtype = DTYPE_MAP[metadata["dtype"]]
        segment_key = metadata["key"]
        mmap_tensor = torch.from_file(file_path, shared=shared, size=size, dtype=dtype)
        
        if segment_key not in mmap_shared_group:
            mmap_shared_group[segment_key] = []
            
        mmap_shared_group[segment_key].append(mmap_tensor)
    
    return mmap_shared_group, num_samples


class BaseSegmentFetcher:
    def __init__(self, segments: Seg):
        self.segments = segments
        self.cum_lengths = []
        total = 0
        for seg in segments:
            total += len(seg)
            self.cum_lengths.append(total)
        self.total_length = total if segments else 0

    def fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        if not (0 <= begin_idx < self.total_length and 0 <= end_idx <= self.total_length):
            raise ValueError("begin_idx or end_idx out of bounds")
        if begin_idx >= end_idx:
            return torch.tensor([], dtype=torch.long)
        
        # fix the range index bug
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
    def __init__(self, muti_segments: MultiSeg):
        self.muti_keys = list(muti_segments.keys())
        self.muti_fetchers = {
            key: BaseSegmentFetcher(segments)
            for key, segments in muti_segments.items()
        }
        
    def key_fetch(self, begin_idx: int, end_idx: int, keys: Union[str, List[str]]) -> Union[Tensor, Seg]:
        fetch_dict = {} 
        keys = [keys] if isinstance(keys, str) else keys
        
        for key in keys:
            fetcher = self.muti_fetchers[key]
            fetch_tensor = fetcher.fetch_data(begin_idx, end_idx)
            fetch_dict[key] = fetch_tensor

        return fetch_dict if len(keys) > 1 else fetch_dict[keys[0]]
    
    def fetch_data(self, begin_idx: int, end_idx: int) -> Union[Tensor, Seg]:
        return self.key_fetch(begin_idx, end_idx, self.muti_keys)


class BaseDataset(Dataset, ABC):
    def __init__(self, window_size: int, stride: int, share_memory: bool=False):
        super().__init__()
        self.segments: MultiSeg = {}
        self.window_size = window_size
        self.stride = stride
        self.total_samples = None

    def load(self, load_path: Union[str, List[str]]):
        paths = [load_path] if isinstance(load_path, str) else load_path
        self.segments, self.total_samples = load_mmap_files(paths)
        self.fetcher = MultiSegmentFetcher(self.segments)
        
    def get_index(self, index: int) -> int:
        begin_idx = min(index * self.stride, self.total_samples - self.window_size - 1)
        end_idx = begin_idx + self.window_size
        
        return begin_idx, end_idx
        
    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        raise NotImplementedError
        
    def __len__(self) -> int:
        assert self.total_samples is not None
        if self.total_samples <= self.window_size:
            return 0
        return self.total_samples // self.stride + 1
    

class SeqDataset(BaseDataset):
    def __init__(self, window_size: int, stride: int):
        super().__init__(window_size, stride)
        self.fetcher = MultiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, "sequence")
    
    def __getitem__(self, index):
        # fix the range index bug
        begin_idx, end_idx = self.get_index(index)
        
        x = self._fetch_data(begin_idx, end_idx).to(dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1).to(dtype=torch.long)
        
        return {"input_ids": x, "target_ids": y}
    
    
class SftDataset(BaseDataset):
    def __init__(self, window_size: int, stride: int):
        super().__init__(window_size, stride)
        self.fetcher = MultiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)
    
    def __getitem__(self, index):
        begin_idx, end_idx = self.get_index(index)
        
        x = self._fetch_data(begin_idx, end_idx, "sequence").to(dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1, "sequence").to(dtype=torch.long)
        loss_mask = self._fetch_data(begin_idx + 1, end_idx + 1, "loss_mask").to(dtype=torch.bool)
        
        return {"input_ids": x, "target_ids": y, "loss_mask": loss_mask}


class DpoDataset(BaseDataset):
    def __init__(self, window_size: int, stride: int):
        super().__init__(window_size, stride)
        self.fetcher = MultiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)

    def __getitem__(self, index: int):
        begin_idx, end_idx = self.get_index(index)
        
        chosen = self._fetch_data(begin_idx, end_idx, "chosen").to(dtype=torch.long)
        rejected = self._fetch_data(begin_idx, end_idx, "rejected").to(dtype=torch.long)
        chosen_mask = self._fetch_data(begin_idx, end_idx, "chosen_mask").to(dtype=torch.bool)
        rejected_mask = self._fetch_data(begin_idx, end_idx, "rejected_mask").to(dtype=torch.bool)

        return {"chosen": chosen, "rejected": rejected, "chosen_mask": chosen_mask, "rejected_mask": rejected_mask}


class PpoDataset(BaseDataset):
    def __init__(self, window_size: int, stride: int):
        super().__init__(window_size, stride)
        self.fetcher = MultiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)
    
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        begin_idx, end_idx = self.get_index(index)

        input_ids =  self._fetch_data(begin_idx, end_idx, "input_ids"),
        actions = self._fetch_data(begin_idx, end_idx, "actions"),
        logprobs = self._fetch_data(begin_idx, end_idx, "logprobs"),
        rewards =  self._fetch_data(begin_idx, end_idx, "rewards")
        
        return {"input_ids": input_ids, "actions": actions, "logprobs": logprobs, "rewards": rewards}
    

class DatasetLoader:
    @staticmethod       
    def load(
        train_type: Literal["seq", "sft", "dpo"],
        load_path: Union[str, List[str]],
        window_size: int, 
        stride: Optional[int] = None,
        **kwargs
        ) -> BaseDataset:
        if stride is None:
            stride = window_size
        
        dataset_router: Dict[str, Callable[[int], BaseDataset]] = {
            "seq": lambda window_size: SeqDataset(window_size, stride),
            "sft": lambda window_size: SftDataset(window_size, stride),
            "dpo": lambda window_size: DpoDataset(window_size, stride),
        }
        dataset = dataset_router[train_type](window_size)
        dataset.load(load_path)
        
        return dataset
