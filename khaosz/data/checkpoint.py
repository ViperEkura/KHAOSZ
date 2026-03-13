import json
import safetensors.torch as st
import torch.distributed as dist

from pathlib import Path
from typing import Dict, Any
from khaosz.parallel.setup import get_rank


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