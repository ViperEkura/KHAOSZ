import torch
import bisect
import pickle as pkl
from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from typing import Callable, List, Dict, Literal, Optional, Union
  
MutiSeg = Dict[str, List[Tensor]]
Seg = Dict[str, Tensor]

def load_pkl_files(paths: List[str]):
    segments: MutiSeg = {}
    total_samples = 0

    for path in paths:
        with open(path, "rb") as f:
            pkl_file: Seg = pkl.load(f)
        for key, value in pkl_file.items():
            if key not in segments:
                segments[key] = []
            segments[key].append(value)
        first_key = list(pkl_file.keys())[0]
        total_samples += pkl_file[first_key].numel()
    
    return segments, total_samples


class BaseSegmentFetcher:
    def __init__(self, segments: List[Tensor]):
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
    

class MutiSegmentFetcher:
    def __init__(self, muti_segments: MutiSeg):
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
    def __init__(self, chunk_size: int, step_size: int):
        super().__init__()
        self.segments: MutiSeg = {}
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.total_samples = None

    def save(self, save_path: str):
        keys = list(self.segments.keys())
        if not keys:
            return
        
        first_item = self.segments[keys[0]]
        segment_size = len(first_item)
        
        for i in range(segment_size):
            formated_segment = {key: self.segments[key][i] for key in keys}
            pkl.dump(formated_segment, open(f"{save_path}_{i}.pkl", "wb"))
    
    def load(self, load_path: Union[str, List[str]]):
        paths = [load_path] if isinstance(load_path, str) else load_path
        self.segments, self.total_samples = load_pkl_files(paths)
        self.fetcher = MutiSegmentFetcher(self.segments)
        
    def get_index(self, index: int) -> int:
        begin_idx = min(index * self.step_size, self.total_samples - self.chunk_size - 1)
        end_idx = begin_idx + self.chunk_size
        
        return begin_idx, end_idx
        
    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        raise NotImplementedError
        
    def __len__(self) -> int:
        assert self.total_samples is not None
        if self.total_samples <= self.chunk_size:
            return 0
        return self.total_samples // self.step_size + 1
    

class SeqDataset(BaseDataset):
    def __init__(self, chunk_size: int, step_size: int):
        super().__init__(chunk_size, step_size)
        self.fetcher = MutiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, "sequence")
    
    def __getitem__(self, index):
        # fix the range index bug
        begin_idx, end_idx = self.get_index(index)
        
        x = self._fetch_data(begin_idx, end_idx).to(dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1).to(dtype=torch.long)
        
        return {"input_ids": x, "target_ids": y}
    
    
class SftDataset(BaseDataset):
    def __init__(self, chunk_size: int, step_size: int):
        super().__init__(chunk_size, step_size)
        self.fetcher = MutiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)
    
    def __getitem__(self, index):
        begin_idx, end_idx = self.get_index(index)
        
        x = self._fetch_data(begin_idx, end_idx, "sequence").to(dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1, "sequence").to(dtype=torch.long)
        loss_mask = self._fetch_data(begin_idx + 1, end_idx + 1, "loss_mask").to(dtype=torch.bool)
        
        return {"input_ids": x, "target_ids": y, "loss_mask": loss_mask}


class DpoDataset(BaseDataset):
    def __init__(self, chunk_size: int, step_size: int):
        super().__init__(chunk_size, step_size)
        self.fetcher = MutiSegmentFetcher(self.segments)

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
    def __init__(self, chunk_size: int, step_size: int):
        super().__init__(chunk_size, step_size)
        self.fetcher = MutiSegmentFetcher(self.segments)

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
        max_len: int, 
        step_size: Optional[int] = None,
        **kwargs
        ) -> BaseDataset:
        if step_size is None:
            step_size = max_len
        
        dataset_router: Dict[str, Callable[[int], BaseDataset]] = {
            "seq": lambda max_len: SeqDataset(max_len, step_size),
            "sft": lambda max_len: SftDataset(max_len, step_size),
            "dpo": lambda max_len: DpoDataset(max_len, step_size),
        }
        dataset = dataset_router[train_type](max_len)
        dataset.load(load_path)
        
        return dataset


class ResumeableRandomSampler(Sampler[int]):
    def __init__(self, data_source, start_epoch=0, start_iter=0, seed=42):
        self.num_samples = len(data_source)
        self.epoch = start_epoch
        self.iter = start_iter
        
        generator = torch.Generator()
        generator.manual_seed(seed)

        # consume  previous epochs
        for _ in range(start_epoch):
            torch.randperm(self.num_samples, generator=generator)
        
        self.generator = generator
        self._indices = None
    
    def _get_indices(self):
        current_epoch_indices = torch.randperm(self.num_samples, generator=self.generator).tolist()
        self._indices = current_epoch_indices[self.iter % self.num_samples:]
    
    def __iter__(self):
        if self._indices is None:
            self._get_indices()
        
        for i in self._indices:
            self.iter += 1
            yield i
        
        self.epoch += 1
        self._indices = None
    
    def __len__(self):
        if self._indices is None:
            self._get_indices()
        return len(self._indices)