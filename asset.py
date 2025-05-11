import torch
from torch import Tensor
from typing import Dict, List, Callable, Tuple


class VectorAssets:
    def __init__(self, file_path=None):
        self.data: List[Dict[str, Tensor]] = []
        if file_path is not None:
            self.load(file_path)
        
    def add_vector(self, key: str, vector_data: Tensor):
        self.data.append({
            "key": key,
            "vector": vector_data
        })
        
    def delete_vector(self, key: str):
        for elm in self.data:
            if elm["key"] == key:
                self.data.remove(elm)    
    
    def simliarity(self, processor: Callable, input_str: str, top_k: int):        
        top_k_clip = min(top_k, len(self.data))
        top_k_data: List[Tuple[str, int]] = []
        inoput_tensor = processor(input_str)
        
        for elm in self.data:
            key, vector = elm["key"], elm["vector"]
            sim = torch.dot(inoput_tensor, vector) / (torch.norm(inoput_tensor) * torch.norm(vector))
            top_k_data.append((key, sim.item()))
            
            if len(top_k_data) >= top_k_clip:
                break
        
        return top_k_data
    
    def load(file_path):
        pass