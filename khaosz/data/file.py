import os
import h5py
import numpy as np
import torch

from pathlib import Path
from torch import Tensor
from typing import Dict, List, Tuple


def save_h5(file_path: str, tensor_group: Dict[str, List[Tensor]]):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, 'w') as f:
        for key, tensors in tensor_group.items():
            grp = f.create_group(key)
            grp.attrs['num_tensors'] = len(tensors)
            
            for idx, tensor in enumerate(tensors):
                arr = tensor.cpu().numpy()
                dset = grp.create_dataset(
                    f'data_{idx}',
                    data=arr
                )
                dset.attrs['numel'] = tensor.numel()

def load_h5(file_path: str) -> Tuple[Dict[str, List[Tensor]], int]:
    tensor_group: Dict[str, List[Tensor]] = {}
    total_samples = 0

    root_path = Path(file_path)
    h5_files = list(root_path.rglob("*.h5")) + list(root_path.rglob("*.hdf5"))
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            for key in f.keys():
                grp = f[key]
                dsets = []
                for dset_name in grp.keys():
                    dset = grp[dset_name]
                    dsets.append(torch.from_numpy(dset[:]).share_memory_())
                    total_samples += dset.attrs.get('numel', np.prod(dset.shape))
                tensor_group[key] = dsets

    num_keys = max(len(tensor_group), 1)
    sample_per_key = total_samples // num_keys

    return tensor_group, sample_per_key