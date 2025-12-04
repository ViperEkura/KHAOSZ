import pickle as pkl
import matplotlib.pyplot as plt
import safetensors.torch as st
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Self, Union
from pathlib import Path

from khaosz.data.tokenizer import BpeTokenizer
from khaosz.config.model_config import ModelConfig
from khaosz.model.transformer import Transformer

@dataclass
class BaseModelIO:
    """Base class for model I/O operations."""
    
    model: Optional[nn.Module] = field(
        default=None,
        metadata={"help": "Transformer model."}
    )
    tokenizer: BpeTokenizer = field(
        default_factory=BpeTokenizer,
        metadata={"help": "Tokenizer for the model."}
    )
    config: ModelConfig = field(
        default_factory=ModelConfig,
        metadata={"help": "Transformer model configuration."}
    )
    
    def _get_file_paths(self, directory: Union[str, Path]) -> dict[str, Path]:
        """Get standardized file paths for model components."""
        dir_path = Path(directory)
        return {
            "model": dir_path / "model.safetensors",
            "config": dir_path / "config.json", 
            "tokenizer": dir_path / "tokenizer.json"
        }
    
    def save_components(self, save_dir: Union[str, Path]):
        """Save core model components."""
        paths = self._get_file_paths(save_dir)
        paths["model"].parent.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            st.save_file(self.model.state_dict(), str(paths["model"]))
        self.config.save(str(paths["config"]))
        self.tokenizer.save(str(paths["tokenizer"]))
    
    def load_components(self, load_dir: Union[str, Path]) -> Self:
        """Load core model components."""
        paths = self._get_file_paths(load_dir)
        
        self.config.load(str(paths["config"]))
        self.tokenizer.load(str(paths["tokenizer"]))
        
        if paths["model"].exists():
            state_dict = st.load_file(str(paths["model"]))
            if self.model is None:
                self.model = Transformer(self.config)
            self.model.load_state_dict(state_dict)
        
        return self
    
    def to(self, *args, **kwargs) -> Self:
        """Move model to device."""
        if self.model is not None:
            self.model.to(*args, **kwargs)
        return self


@dataclass
class ModelParameter(BaseModelIO):
    """Container for model parameters with serialization capabilities."""
    
    def save(self, save_dir: Union[str, Path]):
        self.save_components(save_dir)
    
    def load(self, load_dir: Union[str, Path]) -> Self:
        return self.load_components(load_dir)


@dataclass
class Checkpoint(BaseModelIO):
    """Extended model parameters with training state."""
    
    optimizer_state: Dict[str, Any] = field(
        default=None,
        metadata={"help": "Optimizer state."}
    )
    scheduler_state: Dict[str, Any] = field(
        default=None,
        metadata={"help": "Sampler state."}
    )
    loss_list: List[float] = field(
        default_factory=list,
        metadata={"help": "List of training losses."}
    )
    epoch: int = field(
        default=0,
        metadata={"help": "Current epoch."}
    )
    batch_iter: int = field(
        default=0,
        metadata={"help": "Current iteration."}
    )
    
    def _get_training_paths(self, directory: Union[str, Path]) -> dict[str, Path]:
        dir_path = Path(directory)
        return  {
            "loss_plot": dir_path / "loss_plot.png",
            "training_state": dir_path / "training_state.pkl"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "epoch": self.epoch,
            "batch_iter": self.batch_iter,
            "loss_list": self.loss_list,
        }
    
    def from_dict(self, data: Dict[str, Any]) -> Self:
        self.optimizer_state = data["optimizer_state"]
        self.scheduler_state = data["scheduler_state"]
        self.epoch = data["epoch"]
        self.batch_iter = data["batch_iter"]
        self.loss_list = data["loss_list"]

    def save_training_state(self, save_dir: Union[str, Path]):
        paths = self._get_training_paths(save_dir)
        
        # Save loss plot
        self._plot_loss(str(paths["loss_plot"]))
        
        # Save training state
        with open(str(paths["training_state"]), "wb") as f:
            pkl.dump(self.to_dict(), f)
    
    def load_training_state(self, load_dir: Union[str, Path]) -> Self:
        paths = self._get_training_paths(load_dir)
        
        # Load training state
        with open(str(paths["training_state"]), "rb") as f:
            train_state = pkl.load(f)
        
        self.from_dict(train_state)
        
        return self
    
    def _plot_loss(self, save_path: str): 
        """Plot and save loss curve."""
        if not self.loss_list:
            return
        
        batch_iter = len(self.loss_list)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_list)
        plt.title(f"Training Loss - Iteration {batch_iter}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(save_path, dpi=30, bbox_inches="tight")
        plt.close()
    
    def save(self, save_dir: Union[str, Path]):
        """Save complete checkpoint."""
        self.save_components(save_dir)
        self.save_training_state(save_dir)
    
    def load(self, load_dir: Union[str, Path]) -> Self:
        """Load complete checkpoint."""
        self.load_components(load_dir)
        self.load_training_state(load_dir)
        return self


class ParameterLoader:
    """Factory class for loading model parameters or checkpoints."""
    
    @staticmethod
    def load(load_dir: Union[str, Path]) -> Union[ModelParameter, Checkpoint]:
        """Load either ModelParameter or Checkpoint based on directory contents."""
        load_dir = Path(load_dir)
        
        # Check for training-specific files
        loss_file = load_dir / "loss.pkl"
        has_training_data = loss_file.exists()
        
        # Create appropriate instance
        if has_training_data:
            checkpoint = Checkpoint()
            checkpoint.load(str(load_dir))
            return checkpoint
        else:
            params = ModelParameter()
            params.load(str(load_dir))
            return params
    
    @staticmethod
    def create_checkpoint(
        model: nn.Module, 
        tokenizer: BpeTokenizer,
        config: ModelConfig,
        loss_list: Optional[list[float]] = None,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> Checkpoint:
        """Convenience method to create a training checkpoint."""
        return Checkpoint(
            model=model,
            tokenizer=tokenizer,
            config=config,
            loss_list=loss_list or [],
            optimizer_state=optimizer
        )