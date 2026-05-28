import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import safetensors.torch as st
import torch
import torch.distributed as dist

from astrai.parallel.setup import get_rank

_META_FILE = "meta.json"
_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "model.safetensors"


def save_safetensors(state_dict: dict, path: str | Path):
    st.save_file(state_dict, str(path))


def load_safetensors(path: str | Path) -> dict:
    return st.load_file(str(path))


def save_json(data: dict, path: str | Path):
    with open(str(path), "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(str(path), "r") as f:
        return json.load(f)


def save_torch(obj: Any, path: str | Path):
    torch.save(obj, str(path))


def load_torch(path: str | Path, broadcast: bool = False) -> Any:
    if not broadcast or not dist.is_initialized():
        return torch.load(str(path), map_location="cpu", weights_only=False)

    path = Path(path)
    rank = get_rank()

    if rank == 0:
        with open(path, "rb") as f:
            raw = f.read()
        data_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        num_bytes = torch.tensor([len(raw)], dtype=torch.long)
    else:
        num_bytes = torch.tensor([0], dtype=torch.long)

    dist.broadcast(num_bytes, src=0)

    if rank != 0:
        data_tensor = torch.empty(num_bytes.item(), dtype=torch.uint8)

    dist.broadcast(data_tensor, src=0)

    buf = io.BytesIO(data_tensor.numpy().tobytes())
    return torch.load(buf, map_location="cpu", weights_only=False)


def save_model(config: dict, state_dict: dict, save_directory: str):
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    save_json(config, save_path / _CONFIG_FILE)
    save_safetensors(state_dict, save_path / _WEIGHTS_FILE)


def load_model_config(save_directory: str) -> dict:
    return load_json(Path(save_directory) / _CONFIG_FILE)


def load_model_weights(save_directory: str) -> dict:
    return load_safetensors(Path(save_directory) / _WEIGHTS_FILE)


def _get_meta(save_path: Path) -> dict:
    meta = {}
    if get_rank() == 0:
        meta = load_json(save_path / _META_FILE)
    if dist.is_initialized():
        meta_list = [meta]
        dist.broadcast_object_list(meta_list, src=0)
        meta = meta_list[0]
    return meta


def _load_state_dict(save_path: Path, broadcast: bool = False) -> dict:
    if not broadcast or not dist.is_initialized():
        return load_safetensors(save_path / _WEIGHTS_FILE)

    rank = get_rank()
    if rank == 0:
        state_dict = load_safetensors(save_path / _WEIGHTS_FILE)
        specs: List[Tuple[str, List[int], str]] = [
            (k, list(state_dict[k].shape), str(state_dict[k].dtype).split(".")[-1])
            for k in sorted(state_dict)
        ]
    else:
        state_dict = {}
        specs = []

    specs_list = [specs]
    dist.broadcast_object_list(specs_list, src=0)
    specs = specs_list[0]

    for key, shape, dtype_name in specs:
        dtype = getattr(torch, dtype_name)
        if rank != 0:
            tensor = torch.empty(shape, dtype=dtype, device="cpu")
        else:
            tensor = state_dict[key].contiguous().cpu()
        dist.broadcast(tensor, src=0)
        if rank != 0:
            state_dict[key] = tensor
    return state_dict


@dataclass
class Checkpoint:
    state_dict: Dict[str, Any] = field(default_factory=dict)
    epoch: int = 0
    iteration: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if get_rank() != 0:
            return

        meta = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **self.meta,
        }
        save_json(meta, save_path / _META_FILE)
        save_safetensors(self.state_dict, save_path / _WEIGHTS_FILE)
        for key, value in self.extra.items():
            save_torch(value, save_path / f"{key}.pt")

    @classmethod
    def load(cls, save_dir: str, broadcast: bool = False) -> "Checkpoint":
        save_path = Path(save_dir)

        meta = _get_meta(save_path)
        state_dict = _load_state_dict(save_path, broadcast=broadcast)

        extra = {}
        for f in sorted(save_path.iterdir()):
            if f.suffix == ".pt":
                extra[f.stem] = load_torch(f, broadcast=broadcast)

        return cls(
            state_dict=state_dict,
            epoch=meta.get("epoch", 0),
            iteration=meta.get("iteration", 0),
            extra=extra,
        )
