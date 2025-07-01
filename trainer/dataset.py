import torch
import pickle as pkl
from abc import ABC, abstractmethod
from torch import device
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Union


class BaseDataset(Dataset, ABC):
    def __init__(self, segment_length: int, device: device = device('cuda')):
        super().__init__()
        self.data = None
        self.segment_length = segment_length
        self.total_samples = 0
        self.device = device

    def save(self, save_path: str):
        with open(save_path, "wb") as f:
            pkl.dump(self.data, f)
    @abstractmethod
    def load(self, load_path: Union[str, List[str]]):
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        pass
    
    def __len__(self) -> int:
        assert self.total_samples // self.segment_length > 0
        return self.total_samples // self.segment_length



class SeqDataset(BaseDataset):
    def __init__(self, segment_length , device=device('cuda')):
        super().__init__(segment_length, device)
        self.data = torch.tensor([])
        
    def load(self, load_path: Union[str, List[str]]):
        sequences = []
        
        if isinstance(load_path, list):
            for path in load_path:
                with open(path, "rb") as f:
                    file = pkl.load(f)
                sequences.append(file)
        elif isinstance(load_path, str):
            with open(load_path, "rb") as f:
                file = pkl.load(f)
            sequences.append(file)
        else:
            raise TypeError("load_path: str | list[str]")
        
        self.data = torch.cat(sequences).to(device="cpu", dtype=torch.int32)
        self.total_samples = self.data.numel()
        
    def __getitem__(self, index):
        begin_idx = index * self.segment_length 
        end_idx = min(begin_idx + self.segment_length, self.total_samples - 1)
        
        x = self.data[begin_idx:end_idx].to(device=self.device, dtype=torch.long)
        y = self.data[begin_idx + 1:end_idx + 1].to(device=self.device, dtype=torch.long)
        
        return x, y
    
    
class SftDataset(BaseDataset):
    def __init__(self, segment_length, device=device('cuda')):
        super().__init__(segment_length, device)
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
        begin_idx = index * self.segment_length 
        end_idx = min(begin_idx + self.segment_length, self.total_samples - 1)
        
        x = self.data["sequence"][begin_idx:end_idx].to(device=self.device, dtype=torch.long)
        y = self.data["sequence"][begin_idx + 1:end_idx + 1].to(device=self.device, dtype=torch.long)
        loss_mask = self.data["mask"][begin_idx + 1:end_idx + 1].to(device=self.device, dtype=torch.bool)
        
        return x, y, loss_mask



class DpoDataset(BaseDataset):
    def __init__(self, segment_length: int, device=device("cuda")):
        super().__init__(segment_length, device)
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
        start_idx = index * self.segment_length
        end_idx = min(start_idx + self.segment_length, self.total_samples)
        
        chosen_segment = self.data["chosen"][start_idx:end_idx].to(device=self.device, dtype=torch.long)
        rejected_segment = self.data["rejected"][start_idx:end_idx].to(device=self.device, dtype=torch.long)
        chosen_mask = self.data["chosen_mask"][start_idx:end_idx].to(device=self.device, dtype=torch.bool)
        rejected_mask = self.data["rejected_mask"][start_idx:end_idx].to(device=self.device, dtype=torch.bool)
        
        return chosen_segment, rejected_segment, chosen_mask, rejected_mask
