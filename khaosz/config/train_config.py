from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from torch.utils.data import Dataset
from torch.optim import Optimizer

if TYPE_CHECKING:
    from khaosz.trainer.strategy import BaseStrategy


@dataclass
class TrainConfig:
    
    strategy: "BaseStrategy" = field(
        default=None,
        metadata={"help": "Training strategy."}
    )
    dataset: Dataset = field(
        default=None,
        metadata={"help": "Dataset for training."}
    )
    optimizer: Optimizer = field(
        default=None,
        metadata={"help": "Optimizer for training."}
    )
    checkpoint_dir: str = field(
        default="./checkpoint",
        metadata={"help": "Checkpoint directory."}
    )
    n_epoch: int = field(
        default=1,
        metadata={"help": "Number of epochs for training."}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for training."}
    )
    checkpoint_interval: int = field(
        default=5000,
        metadata={"help": "Number of iterations between checkpoints."}
    )
    accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of iterations between steps."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm."}
    )
    random_seed: int = field(
        default=3407,
        metadata={"help": "Random seed."}
    )
    num_workers: int = field(
        default=0,
        metadata={"help": "Number of workers for dataloader."}
    )
    prefetch_factor: Optional[int] = field(
        default=None,
        metadata={"help": "Prefetch factor for dataloader."}
    )
    pin_memory: bool = field(
        default=False,
        metadata={"help": "Pin memory for dataloader."}
    )