from dataclasses import dataclass, field, fields
from typing import Optional, Self, TYPE_CHECKING
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from khaosz.config import Checkpoint
from khaosz.data import ResumableDistributedSampler
from khaosz.trainer.schedule import BaseScheduler, SchedulerFactory
from khaosz.parallel.utils import get_world_size, get_rank

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
    
    wolrd_size: int = field(default=1)
    rank: int = field(default=0)
    
    def asdict(self) -> dict:
        return {field.name: getattr(self, field.name) 
                for field in fields(self)}


class TrainContextBuilder:
    def __init__(self, trainer: 'Trainer'):
        self.trainer = trainer
        self._context: TrainContext = None
    
    def with_checkpoint(self, checkpoint: Optional[Checkpoint]) -> Self:
        self._context = TrainContext()
        if checkpoint is None:
            checkpoint = Checkpoint(
                model=self.trainer.parameter.model,
                tokenizer=self.trainer.parameter.tokenizer,
                config=self.trainer.parameter.config,
            )
        else:
            # resume from the assigned checkpoint or assigned iteration
            self._context.epoch = max(checkpoint.epoch, self.trainer.train_config.start_epoch)
            self._context.batch_iter = max(checkpoint.batch_iter, self.trainer.train_config.start_batch)
        
        self._context.checkpoint = checkpoint
        return self
    
    def with_optimizer(self) -> Self:
        if self._context is None:
            raise RuntimeError("Must call with_checkpoint() before with_optimizer()")
        
        optimizer = self.trainer.train_config.optimizer
        
        if self._context.checkpoint and self._context.checkpoint.optimizer_state:
            optimizer.load_state_dict(self._context.checkpoint.optimizer_state)
        
        self._context.optimizer = optimizer
        
        if self._context.checkpoint:
            self._context.checkpoint.optimizer_state = optimizer.state_dict()
        
        return self
    
    def with_scheduler(self) -> Self:
        if not hasattr(self._context, 'optimizer') or self._context.optimizer is None:
            raise RuntimeError("Must call with_optimizer() before with_scheduler()")

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
        resumeable_sampler = ResumableDistributedSampler(
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

        if self.trainer.train_config.nprocs > 1:
            self._context.wolrd_size = get_world_size()
            self._context.rank = get_rank()
            
        return self._context