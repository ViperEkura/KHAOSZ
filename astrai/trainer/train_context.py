from dataclasses import dataclass, field
from typing import Optional, Self

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from astrai.config.train_config import TrainConfig
from astrai.dataset import ResumableDistributedSampler
from astrai.parallel.setup import get_current_device, get_rank, get_world_size
from astrai.serialization import Checkpoint
from astrai.trainer.strategy import BaseStrategy, StrategyFactory


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

    world_size: int = field(default=1)
    rank: int = field(default=0)
    kwargs: dict = field(default_factory=dict)


class TrainContextBuilder:
    def __init__(
        self,
        config: TrainConfig,
    ):
        self.config = config
        self._checkpoint: Optional[Checkpoint] = None

    def with_checkpoint(self, checkpoint: Optional[Checkpoint]) -> Self:
        self._checkpoint = checkpoint
        return self

    def build(self) -> TrainContext:
        context = TrainContext(
            model=self.config.model,
            world_size=get_world_size(),
            rank=get_rank(),
        )

        device = get_current_device()
        context.model = context.model.to(device=device)

        if self.config.nprocs > 1 and self.config.parallel_wrapper:
            context.model = self.config.parallel_wrapper(context.model)

        if self._checkpoint is not None:
            context.epoch = max(self._checkpoint.epoch, self.config.start_epoch)
            context.iteration = max(self._checkpoint.iteration, self.config.start_batch)
            context.model.load_state_dict(self._checkpoint.state_dict)
            context.checkpoint = self._checkpoint
        else:
            context.checkpoint = Checkpoint(
                state_dict=context.model.state_dict(),
            )

        context.optimizer = self.config.optimizer_fn(context.model)
        context.scheduler = self.config.scheduler_fn(context.optimizer)

        cfg = self.config
        sampler_offset = context.iteration * cfg.batch_size
        sampler = ResumableDistributedSampler(
            data_source=cfg.dataset,
            start_epoch=context.epoch,
            start_iter=sampler_offset,
            seed=cfg.random_seed,
        )
        context.dataloader = DataLoader(
            cfg.dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
        )

        context.strategy = StrategyFactory.create(
            model=context.model,
            train_type=self.config.strategy,
            device=device,
            **self.config.extra_kwargs,
        )

        return context
