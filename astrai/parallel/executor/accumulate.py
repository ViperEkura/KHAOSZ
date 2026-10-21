"""Gradient accumulation bookkeeping shared by every executor."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class GradientState:
    def __init__(self, grad_accum_steps: int = 1):
        self.num_steps = max(grad_accum_steps, 1)
        self._window_steps: int = 0
        self._sync_gradients: bool = True

    @property
    def sync_gradients(self) -> bool:
        return self._sync_gradients

    @property
    def window_steps(self) -> int:
        return self._window_steps

    def _do_sync(self, force_sync: bool = False):
        if self._sync_gradients:
            self._window_steps = 0
        self._window_steps += 1
        self._sync_gradients = force_sync or self._window_steps == self.num_steps


class AccumOptimizer:
    def __init__(self, optimizer: Optimizer, gradient_state: GradientState):
        self.optimizer = optimizer
        self.gradient_state = gradient_state

    def step(self, closure=None):
        if self.gradient_state.sync_gradients:
            self.optimizer.step(closure)

    def zero_grad(self):
        if self.gradient_state.sync_gradients:
            self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, d):
        self.optimizer.load_state_dict(d)


class AccumScheduler:
    def __init__(self, scheduler: LRScheduler, gradient_state: GradientState):
        self.scheduler = scheduler
        self.gradient_state = gradient_state

    def step(self):
        if self.gradient_state.sync_gradients:
            self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, d):
        self.scheduler.load_state_dict(d)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
