from dataclasses import dataclass, field, fields
from typing import Optional, Self, TYPE_CHECKING
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from khaosz.core.parameter import Checkpoint
from khaosz.trainer.data_util import RandomSampler

if TYPE_CHECKING:
    from khaosz.trainer.trainer import Trainer


@dataclass
class TrainContext:
    dataloader: DataLoader = field(default=None)
    optimizer: Optimizer = field(default=None)
    sampler: RandomSampler = field(default=None)
    epoch: int = field(default=0)
    current_iter: int = field(default=0)
    loss: float = field(default=0.0)
    checkpoint: Checkpoint = field(default=None)
    
    def asdict(self) -> dict:
        return {field.name: getattr(self, field.name) 
                for field in fields(self)}


class TrainContextBuilder:
    def __init__(self, trainer: 'Trainer'):
        self.trainer = trainer
        self._context = TrainContext(
            dataloader=None,
            optimizer=None,
            sampler=None,
            epoch=0,
            current_iter=0,
            loss=0.0,
            checkpoint=None
        )
    
    def with_checkpoint(self, checkpoint: Optional[Checkpoint]) -> Self:
        if checkpoint is None:
            checkpoint = Checkpoint(
                model=self.trainer.parameter.model,
                tokenizer=self.trainer.parameter.tokenizer,
                config=self.trainer.parameter.config,
                sampler_state=None,
                optimizer_state=None,
                loss_list=[]
            )
        self._context.checkpoint = checkpoint
        return self
    
    def with_sampler(self) -> Self:
        seed = self.trainer.train_config.random_seed
        sampler = RandomSampler(
            data_source=self.trainer.train_config.dataset, 
            seed=seed
        )
        
        if self._context.checkpoint and self._context.checkpoint.sampler_state:
            sampler.load_state_dict(self._context.checkpoint.sampler_state)
        
        self._context.sampler = sampler
        self._context.epoch = sampler.epoch
        self._context.current_iter = sampler.current_iter
        
        if self._context.checkpoint:
            self._context.checkpoint.sampler_state = sampler.state_dict()
        
        return self
    
    def with_optimizer(self) -> Self:
        optimizer = self.trainer.train_config.optimizer
        
        if self._context.checkpoint and self._context.checkpoint.optimizer_state:
            optimizer.load_state_dict(self._context.checkpoint.optimizer_state)
        
        self._context.optimizer = optimizer
        
        if self._context.checkpoint:
            self._context.checkpoint.optimizer_state = optimizer.state_dict()
        
        return self
    
    def with_dataloader(self) -> Self:
        dataloader = DataLoader(
            self.trainer.train_config.dataset, 
            batch_size=self.trainer.train_config.batch_size, 
            sampler=self._context.sampler,
            num_workers=self.trainer.train_config.num_workers,
            pin_memory=self.trainer.train_config.pin_memory,
            prefetch_factor=self.trainer.train_config.prefetch_factor
        )
        self._context.dataloader = dataloader
        return self
    
    def build(self) -> TrainContext:
        return self._context