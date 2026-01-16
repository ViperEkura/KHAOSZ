import os
import json
import torch

from torch import Tensor
from typing import List, Dict, Tuple

class MmapFileHandler:
    """
        json metadata like this:
        
        ```
        [
            {"file_name": "file1.pt", "size": 1000, "key": "key1"},
            {"file_name": "file2.pt", "size": 2000, "key": "key2"}
            ...
        ]
        ```
        files like:
        
        ```
        folder_path:
            - metadata.json
            - file1.pt
            - file2.pt
        ...
        ```
    """
    META_DATA = "metadata.json"

    @staticmethod
    def load(root_path: str, shared: bool=True) -> Tuple[Dict[str, List[Tensor]], int]:
        metadata_list = []
        tensor_group: Dict[str, List[Tensor]] = {}
        
        file_mapper_path = os.path.join(root_path, MmapFileHandler.META_DATA)
        if not os.path.exists(file_mapper_path):
            raise FileNotFoundError(f"File mapper not found: {file_mapper_path}")
        
        with open(file_mapper_path, "r") as f:
            metadata_list = json.load(f)
        
        for metadata in metadata_list:
            file_key = metadata["key"]
            file_name = metadata["file_name"]
            file_path = os.path.join(root_path, file_name)
            elm = torch.load(file_path, map_location="cpu", mmap=shared)
                
            if file_key not in tensor_group:
                tensor_group[file_key] = []
            tensor_group[file_key].append(elm)

        num_samples = sum(metadata["size"] for metadata in metadata_list)
        num_keys = max(len(set(metadata['key'] for metadata in metadata_list)), 1)
        sample_per_key = num_samples // num_keys
        
        return tensor_group, sample_per_key 
    
    @staticmethod
    def save(save_path: str, mmap_shared_group: Dict[str, List[Tensor]]) -> None:
        os.makedirs(save_path, exist_ok=True)
        
        metadata_list = []
        for segment_key, segment_tensors in mmap_shared_group.items():
            for idx, tensor in enumerate(segment_tensors):
                
                try:
                    with open(os.path.join(save_path, f"{segment_key}_{idx}.pt"), "wb") as f:
                        torch.save(tensor.contiguous().cpu(), f)
                except Exception as e:
                    raise RuntimeError(f"Error saving tensor: {e}")   
    
                metadata_list.append({
                    "file_name": f"{segment_key}_{idx}.pt",
                    "size": tensor.numel(),
                    "key": segment_key
                })
        
        metadata_path = os.path.join(save_path, MmapFileHandler.META_DATA)
        
        with open(metadata_path, "w") as f:
            json.dump(metadata_list, f)