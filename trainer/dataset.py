import torch
import bisect
import pickle as pkl
from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Union


class SegmentTensorFetcher:
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
    


class BaseDataset(Dataset, ABC):
    def __init__(self, chunk_size: int, device: str):
        super().__init__()
        self.segments: List = []
        self.chunk_size = chunk_size
        self.total_samples = 0
        self.device = device

    def save(self, save_path: str):
        with open(save_path, "wb") as f:
            pkl.dump(self.segments, f)
    
    @abstractmethod
    def load(self, load_path: Union[str, List[str]]):
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        pass
    
    def __len__(self) -> int:
        assert self.total_samples // self.chunk_size > 0
        return self.total_samples // self.chunk_size



class SeqDataset(BaseDataset):
    def __init__(self, chunk_size , device='cuda'):
        super().__init__(chunk_size, device)
        self.fetcher = SegmentTensorFetcher(self.segments)
        
    def load(self, load_path: Union[str, List[str]]):
        paths = [load_path] if isinstance(load_path, str) else load_path

        for path in paths:
            with open(path, "rb") as f:
                pkl_file: Tensor = pkl.load(f)
            self.total_samples += pkl_file.numel()
            self.segments.append(pkl_file)
        
        self.fetcher = SegmentTensorFetcher(self.segments)

        
    def _fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        return self.fetcher.fetch_data(begin_idx, end_idx)
    
    def __getitem__(self, index):
        begin_idx = index * self.chunk_size 
        end_idx = min(begin_idx + self.chunk_size, self.total_samples - 1)
        
        x = self._fetch_data(begin_idx, end_idx).to(device=self.device, dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1).to(device=self.device, dtype=torch.long)
        
        return x, y
    
    
class SftDataset(BaseDataset):
    def __init__(self, chunk_size, device='cuda'):
        super().__init__(chunk_size, device)
        self.data: Dict[str, Tensor] = {
            "sequence": torch.tensor([]),
            "mask": torch.tensor([])
        }

    def load(self, load_path: Union[str, List[str]]):
        sequences = []
        masks = []
        def load_file(path):
            with open(path, "rb") as f:
                file: Dict[str, Tensor] = pkl.load(f)
            sequences.append(file["sequence"].to(device="cpu", dtype=torch.int32))
            masks.append(file["mask"].to(device="cpu", dtype=torch.bool))
        
        if isinstance(load_path, list):
            for path in load_path:
                load_file(path)
        elif isinstance(load_path, str):
            load_file(load_path)
        else:
            raise TypeError("load_path must be str or list[str]")
    
        self.data = {
            "sequence": torch.cat(sequences),
            "mask": torch.cat(masks)
        }
        
        assert self.data["sequence"].numel() == self.data["mask"].numel()
        self.total_samples = self.data["sequence"].numel()
        
    def __getitem__(self, index):
        begin_idx = index * self.chunk_size 
        end_idx = min(begin_idx + self.chunk_size, self.total_samples - 1)
        
        x = self.data["sequence"][begin_idx:end_idx].to(device=self.device, dtype=torch.long)
        y = self.data["sequence"][begin_idx + 1:end_idx + 1].to(device=self.device, dtype=torch.long)
        loss_mask = self.data["mask"][begin_idx + 1:end_idx + 1].to(device=self.device, dtype=torch.bool)
        
        return x, y, loss_mask



class DpoDataset(BaseDataset):
    def __init__(self, chunk_size: int, device="cuda"):
        super().__init__(chunk_size, device)
        self.data: Dict[str, torch.Tensor] = {
            "chosen": torch.tensor([]),
            "rejected": torch.tensor([]),
            "chosen_mask": torch.tensor([]),
            "rejected_mask": torch.tensor([])
        }

    def load(self, load_path: Union[str, List[str]]):
        chosen_data = []
        rejected_data = []
        chosen_mask = []
        rejected_mask = []
        
        def load_file(path):
            with open(path, "rb") as f:
                file: Dict[str, Tensor] = pkl.load(f)
            chosen_data.append(file["chosen"].to(device="cpu", dtype=torch.int32))
            rejected_data.append(file["rejected"].to(device="cpu", dtype=torch.int32))
            chosen_mask.append(file["chosen_mask"].to(device="cpu", dtype=torch.bool))
            rejected_mask.append(file["rejected_mask"].to(device="cpu",dtype=torch.bool))
        
        if isinstance(load_path, list):
            for path in load_path:
                load_file(path)
        elif isinstance(load_path, str):
            load_file(load_path)
        else:
            raise TypeError("load_path must be str or list[str]")
        
        self.data = {
            "chosen": torch.cat(chosen_data),
            "rejected": torch.cat(rejected_data),
            "chosen_mask": torch.cat(chosen_mask),
            "rejected_mask": torch.cat(rejected_mask)
        }
        
        assert self.data["chosen"].numel() == self.data["rejected"].numel()
        self.total_samples = self.data["chosen"].numel()

    def __getitem__(self, index: int):
        start_idx = index * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        
        chosen_segment = self.data["chosen"][start_idx:end_idx].to(device=self.device, dtype=torch.long)
        rejected_segment = self.data["rejected"][start_idx:end_idx].to(device=self.device, dtype=torch.long)
        chosen_mask = self.data["chosen_mask"][start_idx:end_idx].to(device=self.device, dtype=torch.bool)
        rejected_mask = self.data["rejected_mask"][start_idx:end_idx].to(device=self.device, dtype=torch.bool)
        
        return chosen_segment, rejected_segment, chosen_mask, rejected_mask
