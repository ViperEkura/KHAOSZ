import torch
import pickle

from torch import device
from torch.utils.data import Dataset



class SeqDataset(Dataset):
    def __init__(self, m_len, use_uint16=True, device=device('cuda')):
        super().__init__()
        self.data = list()
        self.seg_size = 0
        self.m_len = m_len
        self.device = device
        self.dtype = torch.uint16 if use_uint16 else torch.uint32

    def save(self, save_path):
        self.seg_size = len(self.data) // self.m_len
        with open(save_path, "wb") as f:
            pickle.dump(self.data, f)

    def load(self, load_path:str | list[str]):
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
        
        x = torch.tensor(self.data[begin_idx:end_idx], device=self.device, dtype=self.dtype)
        y = torch.tensor(self.data[begin_idx + 1:end_idx + 1], device=self.device, dtype=self.dtype)
        
        return x, y

    def __len__(self): 
        return self.seg_size

