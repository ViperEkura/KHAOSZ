import torch
import json
from torch import Tensor
from typing import List, Tuple


class Retriever:
    def __init__(self, file_path=None):
        self.items: List[str] = []
        self.embeddings: List[Tensor] = []
        
        if file_path is not None:
            self.load(file_path)
        
    def add_vector(self, key: str, vector_data: Tensor):
        self.items.append(key)
        self.embeddings.append(vector_data.flatten())
        
    def delete_vector(self, key: str):
        for i in range(len(self.items)-1, -1, -1):
            if self.items[i] == key:
                self.items.pop(i)
                self.embeddings.pop(i)
                
    def similarity(self, input_tensor: Tensor, top_k: int) -> List[Tuple[str, float]]:
        if len(self.items) == 0:
            return []
        
        top_k_clip = min(top_k, len(self.items))
        segment = torch.cat(self.embeddings, dim=0)
        sim_scores = torch.matmul(segment, input_tensor)
        
        top_k_data = [
            (self.items[i], sim_scores[i].item()) 
            for i in sim_scores.topk(top_k_clip).indices.tolist()
        ]
        
        return top_k_data
    
    def save(self, file_path):
        serializable_data = [
            {"key": item,"vector": vec.tolist()}
            for (item, vec) in zip(self.items, self.embeddings)
        ]
        
        with open(file_path, "w") as f:
            json.dump(serializable_data, f)
        

    def load(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            
        self.items = []
        self.embeddings = []
        
        for elm in data:
            key = elm["key"]
            vector = torch.tensor(elm["vector"])
            self.items.append(key)
            self.embeddings.append(vector)

    
    