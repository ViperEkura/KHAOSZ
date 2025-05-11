import torch
import json
from torch import Tensor
from typing import Dict, List, Callable, Tuple


class Retriever:
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
    
    def simliarity(self, processor: Callable, input_str: str, top_k: int) -> List[Tuple[str, float]]:
        top_k_clip = min(top_k, len(self.data))
        top_k_data: List[Tuple[str, float]] = []
        inoput_tensor = processor(input_str)
        
        for elm in self.data:
            key, vector = elm["key"], elm["vector"]
            upper = torch.dot(inoput_tensor, vector)
            lower  = torch.norm(inoput_tensor) * torch.norm(vector)
            sim = upper / lower
            top_k_data.append((key, sim.item()))
            
            if len(top_k_data) >= top_k_clip:
                break
        
        return top_k_data
    
    def save(self, file_path):
        serializable_data = [
            {"key": elm["key"],"vector": elm["vector"].tolist()}
            for elm in self.data
        ]
        
        with open(file_path, "w") as f:
            json.dump(serializable_data, f)
        

    def load(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        
        self.data.clear()
        
        for elm in data:
            key = elm["key"]
            vector = torch.tensor(elm["vector"])
            self.data.append({"key": key, "vector": vector})

    
    