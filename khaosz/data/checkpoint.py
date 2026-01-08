import os
import json
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Optional, Any
from khaosz.parallel.setup import get_rank


class Checkpoint:
    def __init__(
        self,
        optimizer_state_dict: Dict[str, Any],
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
        epoch: int = 0,
        iteration: int = 0,
        metrics: Optional[Dict[str, list]] = None,
    ):
        self.optimizer_state_dict = optimizer_state_dict
        self.scheduler_state_dict = scheduler_state_dict
        self.epoch = epoch
        self.iteration = iteration
        self.metrics = metrics or {}

    def save(
        self,
        save_dir: str,
        save_metric_plot: bool = True,
    ) -> None:
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        rank = get_rank()
        if rank == 0:
            meta = {
                "epoch": self.epoch,
                "iteration": self.iteration,
                "metrics": self.metrics,
            }
            with open(save_path / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            if save_metric_plot and self.metrics:
                self._plot_metrics(str(save_path))

        state_dict = {
            "optimizer": self.optimizer_state_dict,
            "scheduler": self.scheduler_state_dict
        }
        with open(save_path / f"state_dict_rank_{get_rank()}.pt", "wb") as f:
            torch.save(state_dict, f)

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

        with open(save_path / f"state_dict_rank_{get_rank()}.pt", "rb") as f:
            state_dict = torch.load(f)

        return cls(
            optimizer_state_dict=state_dict["optimizer"],
            scheduler_state_dict=state_dict["scheduler"],
            epoch=meta["epoch"],
            iteration=meta["iteration"],
            metrics=meta.get("metrics", {}),
        )

    def _plot_metrics(self, save_dir: str):
        for name, values in self.metrics.items():
            if not values:
                continue
            plt.figure(figsize=(10, 6))
            plt.plot(values, label=name)
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.title(f"Training Metric: {name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
            plt.close()