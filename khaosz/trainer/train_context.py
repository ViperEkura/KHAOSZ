from dataclasses import dataclass, field, fields
from typing import Optional, Self, TYPE_CHECKING
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from khaosz.config import Checkpoint
from khaosz.data import ResumeableRandomSampler
from khaosz.trainer.schedule import BaseScheduler, SchedulerFactory

if TYPE_CHECKING:
    from khaosz.trainer.trainer import Trainer


@dataclass
class TrainContext:
    dataloader: DataLoader = field(default=None)
    optimizer: Optimizer = field(default=None)
    scheduler: BaseScheduler = field(default=None)
    checkpoint: Checkpoint = field(default=None)
    epoch: int = field(default=0)
    batch_iter: int = field(default=0)
    loss: float = field(default=0.0)
    
    def asdict(self) -> dict:
        return {field.name: getattr(self, field.name) 
                for field in fields(self)}


class TrainContextBuilder:
    def __init__(self, trainer: 'Trainer'):
        self.trainer = trainer
        self._context = TrainContext()
    
    def with_checkpoint(self, checkpoint: Optional[Checkpoint]) -> Self:
        if checkpoint is None:
            checkpoint = Checkpoint(
                model=self.trainer.parameter.model,
                tokenizer=self.trainer.parameter.tokenizer,
                config=self.trainer.parameter.config,
            )
        else:
            self._context.epoch = checkpoint.epoch
            self._context.batch_iter = checkpoint.batch_iter
        
        self._context.checkpoint = checkpoint
        return self
    
    def with_optimizer(self) -> Self:
        optimizer = self.trainer.train_config.optimizer
        
        if self._context.checkpoint and self._context.checkpoint.optimizer_state:
            optimizer.load_state_dict(self._context.checkpoint.optimizer_state)
        
        self._context.optimizer = optimizer
        
        if self._context.checkpoint:
            self._context.checkpoint.optimizer_state = optimizer.state_dict()
        
        return self
    
    def with_scheduler(self) -> Self:
        # the build order has any problem ?
        optimizer = self.trainer.train_config.optimizer
        schedule_config = self.trainer.schedule_config
        scheduler = SchedulerFactory.load_scheduler(optimizer, schedule_config)
        
        if self._context.checkpoint and self._context.checkpoint.scheduler_state:
            scheduler.load_state_dict(self._context.checkpoint.scheduler_state)
        
        self._context.scheduler = scheduler
        
        if self._context.checkpoint:
            self._context.checkpoint.scheduler_state = scheduler.state_dict()
        
        return self
    
    def with_dataloader(self) -> Self:
        # fix: change batch level batch_iter to sample level offset
        sampler_offset = self._context.batch_iter * self.trainer.train_config.batch_size
        resumeable_sampler = ResumeableRandomSampler(
            data_source=self.trainer.train_config.dataset,
            start_epoch=self._context.epoch,
            start_iter=sampler_offset,
            seed=self.trainer.train_config.random_seed
        )
        
        dataloader = DataLoader(
            self.trainer.train_config.dataset, 
            batch_size=self.trainer.train_config.batch_size, 
            sampler=resumeable_sampler,
            num_workers=self.trainer.train_config.num_workers,
            pin_memory=self.trainer.train_config.pin_memory,
            prefetch_factor=self.trainer.train_config.prefetch_factor
        )
        self._context.dataloader = dataloader
        return self
    
    def build(self) -> TrainContext:
        return self._context