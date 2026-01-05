import os
import pickle as pkl
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Dict, Optional


class Checkpoint:
    def __init__(
        self,
        optimizer_state: Dict[str, Tensor],
        scheduler_state: Dict[str, Tensor],
        epoch: int = 0,
        iteration: int = 0,
        metrics: Optional[Dict[str, list]] = None,
    ):
        self.optimizer_state = optimizer_state
        self.scheduler_state = scheduler_state
        self.epoch, self.iteration = epoch, iteration
        self.metrics = metrics
    
    def save(self, save_dir: str, save_metric_plot=True) -> None:
        os.makedirs(save_dir, exist_ok=True)
        
        train_state = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "metrics": self.metrics,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
        }
        
        with open(os.path.join(save_dir, "train_state.pkl"), "wb") as f:
            pkl.dump(train_state, f)
            
        if save_metric_plot and self.metrics:
            self._plot_metrics(save_dir)
    
    @classmethod
    def load(cls, save_dir: str) -> "Checkpoint":
        checkpoint_path = os.path.join(save_dir, "train_state.pkl")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
        
        with open(checkpoint_path, "rb") as f:
            train_state = pkl.load(f)
        
        return cls(
            optimizer_state=train_state["optimizer_state"],
            scheduler_state=train_state["scheduler_state"],
            epoch=train_state["epoch"],
            iteration=train_state["iteration"],
            metrics=train_state["metrics"]
        )
    
    def _plot_metrics(self, save_dir: str):
        for metric_name, metric_value in self.metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(metric_value, label=metric_name)
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(save_dir, f"{metric_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()