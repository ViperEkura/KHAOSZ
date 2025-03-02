import torch
import pickle as pkl

from torch import device
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Union



class SeqDataset(Dataset):
    def __init__(self, segment_length , device=device('cuda')):
        super().__init__()
        self.data = list()
        self.segment_length  = segment_length 
        self.total_samples = 0
        self.device = device

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pkl.dump(self.data, f)

    def load(self, load_path: Union[str, List[str]]):
        self.data = list()
        if isinstance(load_path, list):
            for path in load_path:
                with open(path, "rb") as f:
                    file = pkl.load(f)
                self.data.extend(file)
        elif isinstance(load_path, str):
            with open(load_path, "rb") as f:
                self.data = pkl.load(f)
        else:
            raise TypeError("load_path: str | list[str]")
        
        self.total_samples = len(self.data)
        
    def __getitem__(self, index):
        begin_idx = index * self.segment_length 
        end_idx = min(begin_idx + self.segment_length, len(self.data) - 1)
        
        x = torch.tensor(self.data[begin_idx:end_idx], device=self.device)
        y = torch.tensor(self.data[begin_idx + 1:end_idx + 1], device=self.device)
        
        return x, y

    def __len__(self): 
        return self.total_samples // self.segment_length
    
    
class SftDataset(Dataset):
    def __init__(self, segment_length , device=device('cuda')):
        super().__init__()
        self.data: Dict[str, list] = {
            "sequence": [],
            "mask": []
        }
        self.segment_length  = segment_length 
        self.total_samples = 0
        self.device = device

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pkl.dump(self.data, f)

    def load(self, load_path: Union[str, List[str]]):
        self.data = {"sequence": [], "mask": []}
        if isinstance(load_path, list):
            for path in load_path:
                with open(path, "rb") as f:
                    file = pkl.load(f)
                self.data["sequence"].extend(file["sequence"])
                self.data["mask"].extend(file["mask"])
                
        elif isinstance(load_path, str):
            with open(load_path, "rb") as f:
                file = pkl.load(f)
            self.data["sequence"].extend(file["sequence"])
            self.data["mask"].extend(file["mask"])
            
        else:
            raise TypeError("load_path: str | list[str]")
        
        assert len(self.data["sequence"]) == len(self.data["mask"])
        self.total_samples = len(self.data["sequence"])
        
    def __getitem__(self, index):
        begin_idx = index * self.segment_length 
        end_idx = min(begin_idx + self.segment_length, len(self.data) - 1)
        
        x = torch.tensor(self.data["sequence"][begin_idx:end_idx], device=self.device)
        x_mask = torch.tensor(self.data["mask"][begin_idx:end_idx], device=self.device)
        
        y = torch.tensor(self.data["sequence"][begin_idx + 1:end_idx + 1], device=self.device)
        y_mask = torch.tensor(self.data["mask"][begin_idx + 1:end_idx + 1], device=self.device)
        
        return x, y, x_mask, y_mask

    def __len__(self): 
        return self.total_samples // self.segment_length


class DpoDataset(Dataset):
    def __init__(self, segment_length: int, device=device("cuda")):
        super().__init__()
        self.data: Dict[str, list] = {
            "accepted": [],
            "rejected": []
        }
        self.segment_length = segment_length
        self.device = device
        self.total_samples = 0

    def save(self, save_path: str):
        with open(save_path, "wb") as f:
            pkl.dump(self.data, f)

    def load(self, load_path: Union[str, List[str]]):
        self.data = {"accepted": [], "rejected": []}
        
        if isinstance(load_path, list):
            for path in load_path:
                with open(path, "rb") as f:
                    file_data = pkl.load(f)
                self.data["accepted"].extend(file_data["accepted"])
                self.data["rejected"].extend(file_data["rejected"])
        elif isinstance(load_path, str):
            with open(load_path, "rb") as f:
                file_data = pkl.load(f)
            self.data["accepted"].extend(file_data["accepted"])
            self.data["rejected"].extend(file_data["rejected"])
        else:
            raise TypeError("load_path: str | list[str]")
        
        assert len(self.data["accepted"]) == len(self.data["rejected"])
        self.total_samples = len(self.data["accepted"])

    def __getitem__(self, index: int):
        start_idx = index * self.segment_length
        end_idx = min(start_idx + self.segment_length, self.total_samples)
        
        accepted_segment = self.data["accepted"][start_idx:end_idx]
        rejected_segment = self.data["rejected"][start_idx:end_idx]
        
        accepted_tensor = torch.stack(accepted_segment)
        rejected_tensor = torch.stack(rejected_segment)
        
        return accepted_tensor, rejected_tensor

    def __len__(self):
        # lower than totoal_samples
        return self.total_samples // self.segment_length