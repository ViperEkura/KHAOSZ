import os
import h5py
import torch

from pathlib import Path
from torch import Tensor
from typing import Dict, List


def save_h5(file_path: str, file_name: str, tensor_group: Dict[str, List[Tensor]]):
    os.makedirs(file_path, exist_ok=True)
    full_file_path = os.path.join(file_path, f"{file_name}.h5")
    with h5py.File(full_file_path, 'w') as f:
        for key, tensors in tensor_group.items():
            grp = f.create_group(key)
            for idx, tensor in enumerate(tensors):
                arr = tensor.cpu().numpy()
                grp.create_dataset(f'data_{idx}', data=arr)

def load_h5(file_path: str, share_memory=True) -> Dict[str, List[Tensor]]:
    tensor_group: Dict[str, List[Tensor]] = {}

    root_path = Path(file_path)
    h5_files = list(root_path.rglob("*.h5")) + list(root_path.rglob("*.hdf5"))
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            for key in f.keys():
                grp = f[key]
                dsets = []
                for dset_name in grp.keys():
                    dset = grp[dset_name]
                    tensor = torch.from_numpy(dset[:])
                    if share_memory:
                        tensor = tensor.share_memory_()
                    dsets.append(tensor)
            
                if tensor_group.get(key) is None:
                    tensor_group[key] = []
                tensor_group[key].extend(dsets)

    return tensor_group