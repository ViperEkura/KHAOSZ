import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import safetensors.torch as st
import torch
import torch.distributed as dist

from astrai.parallel.setup import get_rank

_META_FILE = "meta.json"
_WEIGHTS_FILE = "model.safetensors"
_MODEL_CONFIG_FILE = "config.json"


def save_safetensors(state_dict: dict, path: str | Path) -> None:
    st.save_file(state_dict, str(path))


def load_safetensors(path: str | Path) -> dict:
    return st.load_file(str(path))


def save_json(data: dict, path: str | Path) -> None:
    with open(str(path), "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(str(path), "r") as f:
        return json.load(f)


def save_torch(obj: Any, path: str | Path) -> None:
    torch.save(obj, str(path))


def load_torch(path: str | Path) -> Any:
    return torch.load(str(path), map_location="cpu", weights_only=False)


@dataclass
class Checkpoint:
    state_dict: Dict[str, Any] = field(default_factory=dict)
    epoch: int = 0
    iteration: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, save_dir: str) -> None:
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
    def load(cls, save_dir: str) -> "Checkpoint":
        save_path = Path(save_dir)

        meta = {}
        if get_rank() == 0:
            meta = load_json(save_path / _META_FILE)

        if dist.is_initialized():
            meta_list = [meta]
            dist.broadcast_object_list(meta_list, src=0)
            meta = meta_list[0]

        state_dict = load_safetensors(save_path / _WEIGHTS_FILE)

        extra = {}
        for f in save_path.iterdir():
            if f.suffix == ".pt":
                extra[f.stem] = load_torch(f)

        return cls(
            state_dict=state_dict,
            epoch=meta.get("epoch", 0),
            iteration=meta.get("iteration", 0),
            extra=extra,
        )


def save_model(config: dict, state_dict: dict, save_directory: str) -> None:
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    save_json(config, save_path / _MODEL_CONFIG_FILE)
    save_safetensors(state_dict, save_path / _WEIGHTS_FILE)


def load_model_config(save_directory: str) -> dict:
    return load_json(Path(save_directory) / _MODEL_CONFIG_FILE)


def load_model_weights(save_directory: str) -> dict:
    return load_safetensors(Path(save_directory) / _WEIGHTS_FILE)
