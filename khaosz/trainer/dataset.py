import torch
import bisect
import pickle as pkl
from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Union

MutiSeg = Dict[str, List[Tensor]]
Seg = Dict[str, Tensor]

def load_pkl_files(paths: List[str]):
    segments: MutiSeg = {}
    total_samples = 0

    for path in paths:
        with open(path, "rb") as f:
            pkl_file: Seg = pkl.load(f)
        for key, value in pkl_file.items():
            segments[key] = value
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
        
        seg_start_idx = bisect.bisect_right(self.cum_lengths, begin_idx - 1)
        seg_end_idx = bisect.bisect_left(self.cum_lengths, end_idx - 1)

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
    def __init__(self, chunk_size: int, device: str):
        super().__init__()
        self.segments: MutiSeg = {}
        self.chunk_size = chunk_size
        self.total_samples = 0
        self.device = device

    def save(self, save_path: str):      
        first_item = self.segments[keys[0]]
        segment_size = len(first_item)
        keys = list(self.segments.keys())
        
        for i in range(segment_size):
            formated_segment = {key: self.segments[key][i] for key in keys}
            pkl.dump(formated_segment, open(f"{save_path}_{i}.pkl", "wb"))
                
    
    def load(self, load_path: Union[str, List[str]]):
        paths = [load_path] if isinstance(load_path, str) else load_path
        self.segments, self.total_samples = load_pkl_files(paths)
        self.fetcher = MutiSegmentFetcher(self.segments)
        
    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError
        
    def __len__(self) -> int:
        assert self.total_samples // self.chunk_size > 0
        return self.total_samples // self.chunk_size
    

class SeqDataset(BaseDataset):
    def __init__(self, chunk_size , device='cuda'):
        super().__init__(chunk_size, device)
        self.fetcher = MutiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, "sequence")
    
    def __getitem__(self, index):
        begin_idx = index * self.chunk_size 
        end_idx = min(begin_idx + self.chunk_size, self.total_samples - 1)
        
        x = self._fetch_data(begin_idx, end_idx).to(device=self.device, dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1).to(device=self.device, dtype=torch.long)
        
        return x, y
    
    
class SftDataset(BaseDataset):
    def __init__(self, chunk_size, device='cuda'):
        super().__init__(chunk_size, device)
        self.fetcher = MutiSegmentFetcher(self.segments)
    
    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)
    
    def __getitem__(self, index):
        begin_idx = index * self.chunk_size 
        end_idx = min(begin_idx + self.chunk_size, self.total_samples - 1)
        
        x = self._fetch_data(begin_idx, end_idx, "sequence").to(device=self.device, dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1, "sequence").to(device=self.device, dtype=torch.long)
        loss_mask = self._fetch_data(begin_idx + 1, end_idx + 1, "mask").to(device=self.device, dtype=torch.bool)
        
        return x, y, loss_mask


class DpoDataset(BaseDataset):
    def __init__(self, chunk_size: int, device="cuda"):
        super().__init__(chunk_size, device)
        self.fetcher = MutiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)

    def __getitem__(self, index: int):
        start_idx = index * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples - 1)
        
        chosen = self._fetch_data(start_idx, end_idx, "chosen").to(device=self.device, dtype=torch.long)
        rejected = self._fetch_data(start_idx, end_idx, "rejected").to(device=self.device, dtype=torch.long)
        chosen_mask = self._fetch_data(start_idx, end_idx, "chosen_mask").to(device=self.device, dtype=torch.bool)
        rejected_mask = self._fetch_data(start_idx, end_idx, "rejected_mask").to(device=self.device, dtype=torch.bool)

        return chosen, rejected, chosen_mask, rejected_mask


class PpoDataset(BaseDataset):
    def __init__(self, chunk_size: int, device="cuda"):
        super().__init__(chunk_size, device)
        self.fetcher = MutiSegmentFetcher(self.segments)

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.fetcher.key_fetch(begin_idx, end_idx, key)
    
    def __getitem__(self, index: int) -> Dict[str, Tensor]:

        begin_idx = index * self.chunk_size
        end_idx = min(begin_idx + self.chunk_size, self.total_samples - 1)
        

        input_ids =  self._fetch_data(begin_idx, end_idx, "input_ids").to(self.device),
        actions = self._fetch_data(begin_idx, end_idx, "actions").to(self.device),
        logprobs = self._fetch_data(begin_idx, end_idx, "logprobs").to(self.device),
        rewards =  self._fetch_data(begin_idx, end_idx, "rewards").to(self.device)
        
        return input_ids, actions, logprobs, rewards
        