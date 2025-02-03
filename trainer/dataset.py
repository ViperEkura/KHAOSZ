import torch
import pickle

from torch import device
from torch.utils.data import Dataset
from typing import List, Dict, Union



class SeqDataset(Dataset):
    def __init__(self, m_len, device=device('cuda')):
        super().__init__()
        self.data = list()
        self.seg_size = 0
        self.m_len = m_len
        self.device = device

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(self.data, f)

    def load(self, load_path: Union[str, List[str]]):
        self.data = list()
        if isinstance(load_path, list):
            for path in load_path:
                with open(path, "rb") as f:
                    file = pickle.load(f)
                self.data.extend(file)
        elif isinstance(load_path, str):
            with open(load_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            raise TypeError("load_path: str | list[str]")
        self.seg_size = len(self.data) // self.m_len
        
    def __getitem__(self, index):
        begin_idx = index * self.m_len
        end_idx = min(begin_idx + self.m_len, len(self.data))
        
        x = torch.tensor(self.data[begin_idx:end_idx], device=self.device)
        y = torch.tensor(self.data[begin_idx + 1:end_idx + 1], device=self.device)
        
        return x, y

    def __len__(self): 
        return self.seg_size


class DpoDataset(Dataset):
    def __init__(self, m_len, device=device('cuda')):
        super().__init__()
        self.data: Dict[str, List[int]] = {
            "prompt": [],
            "response": [],
            "rejected": []
        }
        self.seg_size = 0
        self.m_len = m_len
        self.device = device

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(self.data, f)

    def load(self, load_path: Union[str, List[str]]):
        self.data = list()
        if isinstance(load_path, list):
            for path in load_path:
                with open(path, "rb") as f:
                    file = pickle.load(f)
                self.data["prompt"].extend(file["prompt"])
                self.data["response"].extend(file["response"])
                self.data["rejected"].extend(file["rejected"])
        elif isinstance(load_path, str):
            with open(load_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            raise TypeError("load_path: str | list[str]")
        self.seg_size = len(self.data) // self.m_len
        
    def __getitem__(self, index):
        begin_idx = index * self.m_len
        end_idx = min(begin_idx + self.m_len, len(self.data))
        
        prompt = torch.tensor(self.data["prompt"][begin_idx:end_idx], device=self.device)
        response = torch.tensor(self.data["response"][begin_idx + 1:end_idx + 1], device=self.device)
        rejected = torch.tensor(self.data["rejected"][begin_idx + 1:end_idx + 1], device=self.device)
    
        return prompt, response, rejected

    def __len__(self): 
        return self.seg_size
    