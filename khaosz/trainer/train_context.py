import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from khaosz.data import ResumableDistributedSampler
from khaosz.trainer.checkpoint import Checkpoint
from khaosz.trainer.strategy import StrategyFactory, BaseStrategy
from khaosz.config.train_config import TrainConfig
from khaosz.parallel.utils import get_current_device, get_world_size, get_rank

from dataclasses import dataclass, field
from typing import Optional, Self


@dataclass
class TrainContext:
    model: nn.Module = field(default=None)
    strategy: BaseStrategy = field(default=None)
    dataloader: DataLoader = field(default=None)
    optimizer: Optimizer = field(default=None)
    scheduler: LRScheduler = field(default=None)
    checkpoint: Checkpoint = field(default=None)
    
    epoch: int = field(default=0)
    iteration: int = field(default=0)
    loss: float = field(default=0.0)
    
    wolrd_size: int = field(default=1)
    rank: int = field(default=0)


class TrainContextBuilder:
    def __init__(self, config: TrainConfig):
        self.config = config
        self._context: TrainContext = None
    
    def with_checkpoint(self, checkpoint: Optional[Checkpoint]) -> Self:
        self._context = TrainContext()
        if checkpoint is None:
            checkpoint = Checkpoint(
                optimizer_state=self.config.optimizer.state_dict(),
                scheduler_state=self.config.scheduler.state_dict(),
            )
        else:
            # resume from the assigned checkpoint or assigned iteration
            self._context.epoch = max(checkpoint.epoch, self.config.start_epoch)
            self._context.iteration = max(checkpoint.iteration, self.config.start_batch)
        
        self._context.checkpoint = checkpoint
        return self
    
    def with_optimizer(self) -> Self:
        optimizer = self.config.optimizer
        if self._context is None:
            raise RuntimeError("Must call with_checkpoint() before with_optimizer()")
        
        if self._context.checkpoint and self._context.checkpoint.optimizer_state:
            optimizer.load_state_dict(self._context.checkpoint.optimizer_state)
        
        self._context.optimizer = optimizer
        
        if self._context.checkpoint:
            self._context.checkpoint.optimizer_state = optimizer.state_dict()
        
        return self
    
    def with_scheduler(self) -> Self:
        scheduler = self.config.scheduler
        if self._context.checkpoint and self._context.checkpoint.scheduler_state:
            scheduler.load_state_dict(self._context.checkpoint.scheduler_state)
        
        self._context.scheduler = scheduler
        
        if self._context.checkpoint:
            self._context.checkpoint.scheduler_state = scheduler.state_dict()
        
        return self
    
    def with_dataloader(self) -> Self:
        # fix: change batch level iteration to sample level offset
        config = self.config
        sampler_offset = self._context.iteration * config.batch_size
        resumeable_sampler = ResumableDistributedSampler(
            data_source=config.dataset,
            start_epoch=self._context.epoch,
            start_iter=sampler_offset,
            seed=config.random_seed
        )
        
        dataloader = DataLoader(
            config.dataset, 
            batch_size=config.batch_size, 
            sampler=resumeable_sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor
        )
        self._context.dataloader = dataloader
        return self
    
    def with_strategy(self) -> Self:
        device = get_current_device()
        self._context.strategy = StrategyFactory.load(
            model=self.config.model,
            train_type=self.config.strategy,
            device=device,
            **self.config.kwargs
        )
        return self
    
    def build(self) -> TrainContext:
        self._context.model = self.config.model
        
        if self.config.nprocs > 1:
            self._context.wolrd_size = get_world_size()
            self._context.rank = get_rank()
            
        return self._context