"""Stochastic gradient descent with warm restarts."""

import math

from astrai.optimization.scheduler.base import BaseScheduler, SchedulerFactory


@SchedulerFactory.register("sgdr")
class SGDRScheduler(BaseScheduler):
    """SGDR (Stochastic Gradient Descent with Warm Restarts) scheduler."""

    _required_state_keys = ("warmup_steps", "cycle_length", "min_rate", "t_mult")

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        cycle_length: int,
        min_rate: float = 0.01,
        t_mult: int = 2,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.cycle_length = cycle_length
        self.min_rate = min_rate
        self.t_mult = t_mult

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # warmup
        if self.last_epoch < self.warmup_steps:
            warmup_factor = max(
                self.min_rate, self.last_epoch / max(self.warmup_steps, 1)
            )
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # SGDR
        steps_since_warmup = self.last_epoch - self.warmup_steps

        # 1. Calculate current cycle and position within cycle
        current_cycle_length = self.cycle_length
        total_cycles_length = 0
        cycle_num = 0

        while total_cycles_length + current_cycle_length <= steps_since_warmup:
            total_cycles_length += current_cycle_length
            current_cycle_length *= self.t_mult
            cycle_num += 1

        steps_in_cycle = steps_since_warmup - total_cycles_length

        # 2. Cosine annealing within the current cycle
        cosine_factor = 0.5 * (
            1 + math.cos(math.pi * steps_in_cycle / current_cycle_length)
        )
        learning_rate_factor = self.min_rate + (1 - self.min_rate) * cosine_factor

        return [base_lr * learning_rate_factor for base_lr in self.base_lrs]
