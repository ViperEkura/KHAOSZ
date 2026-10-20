"""Cosine decay with warmup."""

import math
from typing import List

from astrai.optimization.scheduler.base import BaseScheduler, SchedulerFactory


@SchedulerFactory.register("cosine")
class CosineScheduler(BaseScheduler):
    """Cosine decay scheduler with warmup, implemented as PyTorch LRScheduler."""

    _required_state_keys = ("warmup_steps", "lr_decay_steps", "min_rate", "total_steps")

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        lr_decay_steps: int,
        min_rate: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.min_rate = min_rate
        self.total_steps = warmup_steps + lr_decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # warmup
        if self.last_epoch < self.warmup_steps:
            warmup_factor = max(
                self.min_rate, self.last_epoch / max(self.warmup_steps, 1)
            )
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # cosine decay
        decay_progress = (self.last_epoch - self.warmup_steps) / max(
            self.lr_decay_steps, 1
        )
        decay_progress = min(decay_progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        decay_factor = max(self.min_rate, cosine_decay)
        return [base_lr * decay_factor for base_lr in self.base_lrs]
