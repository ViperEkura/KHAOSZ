from torch import nn
from torch.utils.data import Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class TrainConfig:
    # basic setting
    model: nn.Module = field(
        default=None,
        metadata={"help": "Model for training."}
    )
    strategy: str = field(
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
    scheduler: LRScheduler = field(
        default=None,
        metadata={"help": "Scheduler for training."}
    )
    n_epoch: int = field(
        default=1,
        metadata={"help": "Number of epochs for training."}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for training."}
    )
    accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of iterations between steps."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm."}
    )
    
    # checkpoint setting
    start_epoch: int = field(
        default=0,
        metadata={"help": "Start epoch for training."}
    )
    start_batch: int = field(
        default=0,
        metadata={"help": "Start batch iteration for training."}
    )
    checkpoint_dir: str = field(
        default="./checkpoint",
        metadata={"help": "Checkpoint directory."}
    )
    checkpoint_interval: int = field(
        default=5000,
        metadata={"help": "Number of iterations between checkpoints."}
    )
    
    # dataloader setting
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
    
    # distributed training
    nprocs: int = field(
        default=1,
        metadata={"help": "Number of processes for distributed training."}
    )
    backend: str = field(
        default="nccl",
        metadata={"help": "Distributed training backend."}
    )
    master_addr: str = field(
        default="localhost",
        metadata={"help": "Master address for distributed training."}
    )
    master_port: str = field(
        default="29500",
        metadata={"help": "Master port for distributed training."}
    )
    parallel_fn: Optional[Callable] = field(
        default=None,
        metadata={"help": "Parallel function for training."}
    )
    
    # others
    extra_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Other arguments."}
    )
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        required_fields = ["model", "strategy", "dataset", "optimizer", "scheduler"]
        for field_name in required_fields:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} is required.")
