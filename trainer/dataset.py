import torch
import pickle as pkl

from torch import device
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Union



class SeqDataset(Dataset):
    def __init__(self, segment_length , device=device('cuda')):
        super().__init__()
        self.data = torch.tensor([])
        self.segment_length  = segment_length 
        self.total_samples = 0
        self.device = device

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pkl.dump(self.data, f)

    def load(self, load_path: Union[str, List[str]]):
        sequences = []
        
        if isinstance(load_path, list):
            for path in load_path:
                with open(path, "rb") as f:
                    file = pkl.load(f)
                sequences.extend(file)
        elif isinstance(load_path, str):
            with open(load_path, "rb") as f:
                file = pkl.load(f)
            sequences.extend(file)
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

    def __len__(self): 
        return self.total_samples // self.segment_length
    
    
class SftDataset(Dataset):
    def __init__(self, segment_length, device=device('cuda')):
        super().__init__()
        self.data: Dict[str, Tensor] = {
            "sequence": torch.tensor([]),
            "mask": torch.tensor([])
        }
        self.segment_length  = segment_length 
        self.total_samples = 0
        self.device = device

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pkl.dump(self.data, f)

    def load(self, load_path: Union[str, List[str]]):
        sequences = []
        masks = []
        def load_file(path):
            with open(path, "rb") as f:
                file: Dict[str, Tensor] = pkl.load(f)
            sequences.append(file["sequence"].to(dtype=torch.int32))
            masks.append(file["mask"].to(dtype=torch.bool))
        
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

    def __len__(self): 
        return self.total_samples // self.segment_length


class DpoDataset(Dataset):
    def __init__(self, segment_length: int, device=device("cuda")):
        super().__init__()
        self.data: Dict[str, torch.Tensor] = {
            "accepted": torch.tensor([]),
            "rejected": torch.tensor([])
        }
        self.segment_length = segment_length
        self.device = device
        self.total_samples = 0

    def save(self, save_path: str):
        with open(save_path, "wb") as f:
            pkl.dump(self.data, f)

    def load(self, load_path: Union[str, List[str]]):
        accepted_data = []
        rejected_data = []
        
        def load_file(path):
            with open(path, "rb") as f:
                file: Dict[str, Tensor] = pkl.load(f)
            accepted_data.append(file["accepted"].to(dtype=torch.int32))
            rejected_data.append(file["rejected"].to(dtype=torch.int32))
        
        if isinstance(load_path, list):
            for path in load_path:
                load_file(path)
        elif isinstance(load_path, str):
            load_file(load_path)
        else:
            raise TypeError("load_path must be str or list[str]")
        
        self.data = {
            "accepted": torch.cat(accepted_data),
            "rejected": torch.cat(rejected_data)
        }
        
        assert self.data["accepted"].numel() == self.data["rejected"].numel()
        self.total_samples = self.data["accepted"].numel()

    def __getitem__(self, index: int):
        start_idx = index * self.segment_length
        end_idx = min(start_idx + self.segment_length, self.total_samples - 1)
        
        accepted_segment = self.data["accepted"][start_idx:end_idx].to(device=self.device, dtype=torch.long)
        rejected_segment = self.data["rejected"][start_idx:end_idx].to(device=self.device, dtype=torch.long)
        
        return accepted_segment, rejected_segment

    def __len__(self):
        return self.total_samples // self.segment_length