import os
import pickle as pkl
import matplotlib.pyplot as plt

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Dict, Optional


class Checkpoint:
    def __init__(
        self,
        optimizer_state: Optimizer,
        scheduler_state: LRScheduler,
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
            self._plot_metrics()
    
    def load(self, save_dir: str) -> "Checkpoint":
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Checkpoint directory {save_dir} does not exist.")
        
        with open(os.path.join(save_dir, "train_state.pkl"), "rb") as f:
            train_state = pkl.load(f)
            self.epoch = train_state["epoch"]
            self.iteration = train_state["iteration"]
            self.metrics = train_state["metrics"]
            self.optimizer_state = train_state["optimizer_state"]
            self.scheduler_state = train_state["scheduler_state"]
        
        return self
    
    def _plot_metrics(self):
        for metric_name, metric_value in self.metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(metric_value, label=metric_name)
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(f'{metric_name}.png', dpi=150, bbox_inches='tight')
            plt.close()