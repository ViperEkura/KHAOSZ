from dataclasses import dataclass, field
from typing import Optional, Self

import torch.nn as nn
from torch.utils.data import DataLoader

from astrai.config.train_config import TrainConfig
from astrai.dataset import ResumableDistributedSampler
from astrai.model.components.lora import inject_lora
from astrai.parallel.executor import BaseExecutor, ExecutorFactory
from astrai.parallel.setup import get_current_device, get_rank, get_world_size
from astrai.protocols import OptimizerProtocol, SchedulerProtocol
from astrai.serialization import Checkpoint
from astrai.trainer.strategy import BaseStrategy, StrategyFactory


@dataclass
class TrainContext:
    model: nn.Module = field(default=None)
    strategy: BaseStrategy = field(default=None)
    dataloader: DataLoader = field(default=None)
    optimizer: OptimizerProtocol = field(default=None)
    scheduler: SchedulerProtocol = field(default=None)
    checkpoint: Checkpoint = field(default=None)
    config: TrainConfig = field(default=None)
    executor: BaseExecutor = field(default=None)

    epoch: int = field(default=0)
    iteration: int = field(default=0)
    loss: float = field(default=0.0)
    val_dataloader: DataLoader = field(default=None)
    val_loss: float = field(default=0.0)

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
        cfg = self.config
        device = get_current_device()

        executor = ExecutorFactory.create(
            cfg.parallel_mode,
            grad_accum_steps=cfg.grad_accum_steps,
            **cfg.executor_kwargs,
        )

        context = TrainContext(
            model=cfg.model,
            world_size=get_world_size(),
            rank=get_rank(),
            config=cfg,
            executor=executor,
        )

        context.model = context.model.to(device=device)

        if self._checkpoint is not None:
            context.epoch = max(self._checkpoint.epoch, cfg.start_epoch)
            context.iteration = max(self._checkpoint.iteration, cfg.start_batch)
            if self._checkpoint.state_dict:
                context.model.load_state_dict(self._checkpoint.state_dict)
            context.checkpoint = self._checkpoint
        else:
            context.checkpoint = Checkpoint(
                state_dict=context.model.state_dict(),
            )

        if cfg.lora is not None:
            inject_lora(
                context.model,
                r=cfg.lora.r,
                alpha=cfg.lora.alpha,
                target_modules=set(cfg.lora.target_modules),
            )

        context.optimizer = cfg.optimizer_fn(context.model)
        context.scheduler = cfg.scheduler_fn(context.optimizer)

        sampler_offset = context.iteration * cfg.batch_per_device
        sampler = ResumableDistributedSampler(
            data_source=cfg.dataset,
            start_epoch=context.epoch,
            start_iter=sampler_offset,
            seed=cfg.random_seed,
        )
        context.dataloader = DataLoader(
            cfg.dataset,
            batch_size=cfg.batch_per_device,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
        )

        if cfg.val_dataset is not None:
            val_sampler = ResumableDistributedSampler(
                data_source=cfg.val_dataset,
                start_epoch=0,
                start_iter=0,
                seed=cfg.random_seed,
                shuffle=False,
            )
            context.val_dataloader = DataLoader(
                cfg.val_dataset,
                batch_size=cfg.batch_per_device,
                sampler=val_sampler,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                prefetch_factor=cfg.prefetch_factor,
            )

        context.model, context.optimizer, context.dataloader, context.scheduler = (
            executor.prepare(
                context.model,
                context.optimizer,
                context.dataloader,
                context.scheduler,
            )
        )

        context.strategy = StrategyFactory.create(
            model=context.model,
            train_type=cfg.strategy,
            device=device,
            **cfg.extra_kwargs,
        )

        return context
