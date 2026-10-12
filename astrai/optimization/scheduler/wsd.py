"""Warmup-stable-decay learning-rate schedule."""

from .base import BaseScheduler
from .factory import SchedulerFactory


@SchedulerFactory.register("wsd")
class WSDScheduler(BaseScheduler):
    """WSD (Warmup-Stable-Decay) scheduler with sqrt cooldown.

    warmup_steps:  linear warmup from min_rate to 1.0
    stable_steps:   constant at base_lr
    decay_steps:    sqrt decay from base_lr to min_rate
    min_rate:       minimum lr as fraction of base_lr (default 0.0)
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_rate: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_rate = min_rate
        self.total_steps = warmup_steps + stable_steps + decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            factor = max(self.min_rate, self.last_epoch / max(self.warmup_steps, 1))
            return [base_lr * factor for base_lr in self.base_lrs]

        offset = self.last_epoch - self.warmup_steps

        if offset < self.stable_steps:
            return list(self.base_lrs)

        decay_ratio = (offset - self.stable_steps) / max(self.decay_steps, 1)
        decay_ratio = min(decay_ratio, 1.0)
        factor = (1.0 - self.min_rate) * (1.0 - decay_ratio) ** 2 + self.min_rate
        return [base_lr * factor for base_lr in self.base_lrs]

    def state_dict(self):
        state = super().state_dict()
        state.update(
            {
                "warmup_steps": self.warmup_steps,
                "stable_steps": self.stable_steps,
                "decay_steps": self.decay_steps,
                "min_rate": self.min_rate,
                "total_steps": self.total_steps,
            }
        )
        return state

    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict.pop("warmup_steps")
        self.stable_steps = state_dict.pop("stable_steps")
        self.decay_steps = state_dict.pop("decay_steps")
        self.min_rate = state_dict.pop("min_rate")
        self.total_steps = state_dict.pop("total_steps")
        super().load_state_dict(state_dict)
