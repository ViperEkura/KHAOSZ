import os
import h5py
import torch
import json
import safetensors.torch as st
import torch.distributed as dist

from pathlib import Path
from torch import Tensor
from typing import Any, Dict, List
from khaosz.parallel.setup import get_rank

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


class Checkpoint:
    def __init__(
        self,
        state_dict: Dict[str, Any],
        epoch: int = 0,
        iteration: int = 0,
    ):
        self.state_dict = state_dict
        self.epoch = epoch
        self.iteration = iteration

    def save(
        self,
        save_dir: str,
    ) -> None:
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        rank = get_rank()
        if rank == 0:
            meta = {
                "epoch": self.epoch,
                "iteration": self.iteration,
            }
            with open(save_path / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            
            st.save_file(self.state_dict, save_path / f"state_dict.safetensors")

    @classmethod
    def load(
        cls,
        save_dir: str,
    ) -> "Checkpoint":

        rank = get_rank()
        save_path = Path(save_dir)
        
        meta = {}
        if rank == 0:
            with open(Path(save_dir) / "meta.json", "r") as f:
                meta = json.load(f)

        if dist.is_initialized():
            meta_list = [meta]
            dist.broadcast_object_list(meta_list, src=0)
            meta = meta_list[0]

        state_dict = st.load_file(save_path / f"state_dict.safetensors")

        return cls(
            state_dict=state_dict,
            epoch=meta["epoch"],
            iteration=meta["iteration"],
        )