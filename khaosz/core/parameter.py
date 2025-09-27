import pickle as pkl
import matplotlib.pyplot as plt
import safetensors.torch as st
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass, field
from typing import Optional, Self, Union
from pathlib import Path

from khaosz.core.tokenizer import BpeTokenizer
from khaosz.core.transformer import TransformerConfig, Transformer


class BaseModelIO:
    """Base class for model I/O operations."""
    
    def __init__(
        self, 
        model: Optional[nn.Module] = None, 
        tokenizer: Optional[BpeTokenizer] = None,
        config: Optional[TransformerConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer or BpeTokenizer()
        self.config = config or TransformerConfig()
    
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
    
    model: Optional[nn.Module] = field(
        default=None,
        metadata={"help": "Transformer model."}
    )
    tokenizer: BpeTokenizer = field(
        default_factory=BpeTokenizer,
        metadata={"help": "Tokenizer for the model."}
    )
    config: TransformerConfig = field(
        default_factory=TransformerConfig,
        metadata={"help": "Transformer model configuration."}
    )
    
    def save(self, save_dir: Union[str, Path]):
        """Save model parameters."""
        self.save_components(save_dir)
    
    def load(self, load_dir: Union[str, Path]) -> Self:
        """Load model parameters."""
        return self.load_components(load_dir)


@dataclass
class Checkpoint(BaseModelIO):
    """Extended model parameters with training state."""
    
    model: Optional[nn.Module] = field(
        default=None,
        metadata={"help": "Transformer model."}
    )
    tokenizer: BpeTokenizer = field(
        default_factory=BpeTokenizer,
        metadata={"help": "Tokenizer for the model."}
    )
    config: TransformerConfig = field(
        default_factory=TransformerConfig,
        metadata={"help": "Transformer model configuration."}
    )
    loss_list: list[float] = field(
        default_factory=list,
        metadata={"help": "List of training losses."}
    )
    current_iter: int = field(
        default=0,
        metadata={"help": "Current training iteration."}
    )
    optimizer: Optional[optim.Optimizer] = field(
        default=None,
        metadata={"help": "Optimizer state."}
    )
    
    def __post_init__(self):
        # Ensure current_iter matches loss list length if not explicitly set
        if self.current_iter == 0 and self.loss_list:
            self.current_iter = len(self.loss_list)
    
    def _get_training_paths(self, directory: Union[str, Path]) -> dict[str, Path]:
        """Get file paths for training-specific files."""
        paths = self._get_file_paths(directory)
        paths.update({
            "loss_list": paths["model"].parent / "loss.pkl",
            "loss_plot": paths["model"].parent / "loss.png",
            "optimizer": paths["model"].parent / "optimizer.pkl"
        })
        return paths
    
    def save_training_state(self, save_dir: Union[str, Path]):
        """Save training-specific state."""
        paths = self._get_training_paths(save_dir)
        
        # Save loss plot
        self._plot_loss(str(paths["loss_plot"]))
        
        # Save loss list
        with open(str(paths["loss_list"]), "wb") as f:
            pkl.dump(self.loss_list, f)
        
        # Save optimizer state
        if self.optimizer is not None:
            with open(str(paths["optimizer"]), "wb") as f:
                pkl.dump(self.optimizer.state_dict(), f)
    
    def load_training_state(self, load_dir: Union[str, Path]) -> Self:
        """Load training-specific state."""
        paths = self._get_training_paths(load_dir)
        
        # Load loss list
        if paths["loss_list"].exists():
            with open(str(paths["loss_list"]), "rb") as f:
                self.loss_list = pkl.load(f)
            self.current_iter = len(self.loss_list)
        
        # Load optimizer state
        if paths["optimizer"].exists() and self.optimizer is not None:
            with open(str(paths["optimizer"]), "rb") as f:
                optim_state = pkl.load(f)
            self.optimizer.load_state_dict(optim_state)
        
        return self
    
    def _plot_loss(self, save_path: str): 
        """Plot and save loss curve."""
        if not self.loss_list:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_list)
        plt.title(f"Training Loss - Iteration {self.current_iter}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
        config: TransformerConfig,
        loss_list: Optional[list[float]] = None,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Checkpoint:
        """Convenience method to create a training checkpoint."""
        return Checkpoint(
            model=model,
            tokenizer=tokenizer,
            config=config,
            loss_list=loss_list or [],
            optimizer=optimizer
        )


