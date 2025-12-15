import os
import json
import torch

from torch import Tensor
from typing import List, Dict, Tuple

class MmapFileHander:
    """
        json metadata like this:
        
        ```
        [
            {"file_name": "file1.bin", "size": 1000, "dtype": "float32", "key": "key1"},
            {"file_name": "file2.bin", "size": 2000, "dtype": "float32", "key": "key2"}
            ...
        ]
        ```
        files like:
        
        ```
        file_mapper.json
        file1.bin
        file2.bin
        ...
        
        ```
    """

    DTYPE_MAP = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    REVERSE_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}

    @staticmethod
    def load(root_path: str, shared: bool=True) -> Tuple[Dict[str, List[Tensor]], int]:
        metadata_list = []
        mmap_shared_group: Dict[str, List[Tensor]] = {}
        
        file_mapper_path = os.path.join(root_path, "file_mapper.json")
        if not os.path.exists(file_mapper_path):
            raise FileNotFoundError(f"File mapper not found: {file_mapper_path}")
        
        with open(file_mapper_path, "r") as f:
            metadata_list = json.load(f)
        
        for metadata in metadata_list:
            file_path = os.path.join(root_path, metadata["file_name"])
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Binary data file not found: {file_path}")
            
            size = metadata["size"]
            dtype = MmapFileHander.DTYPE_MAP[metadata["dtype"]]
            segment_key = metadata["key"]
            mmap_tensor = torch.from_file(file_path, shared=shared, size=size, dtype=dtype)
            
            if segment_key not in mmap_shared_group:
                mmap_shared_group[segment_key] = []
                
            mmap_shared_group[segment_key].append(mmap_tensor)
        
        num_samples = sum(metadata["size"] for metadata in metadata_list)
        num_keys = len(set(metadata['key'] for metadata in metadata_list))
        
        sample_per_key = num_samples / num_keys
        
        return mmap_shared_group, sample_per_key
    
    @staticmethod
    def save(save_path: str, mmap_shared_group: Dict[str, List[Tensor]]) -> None:
        os.makedirs(save_path, exist_ok=True)
        
        metadata_list = []
        for segment_key, segment_tensors in mmap_shared_group.items():
            for idx, tensor in enumerate(segment_tensors):
                metadata_list.append({
                    "file_name": f"{segment_key}_{idx}.bin",
                    "size": tensor.numel(),
                    "dtype": MmapFileHander.REVERSE_DTYPE_MAP[tensor.dtype],
                    "key": segment_key
                })
                file_path = os.path.join(save_path, f"{segment_key}_{idx}.bin")
                with open(file_path, "wb") as f:
                    f.write(tensor.cpu().numpy().tobytes())
        
        metadata_path = os.path.join(save_path, "file_mapper.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata_list, f)
