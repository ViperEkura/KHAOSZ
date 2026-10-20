"""Base class for training learning-rate schedulers."""

from abc import ABC, abstractmethod
from typing import List, Tuple

from torch.optim.lr_scheduler import LRScheduler

from astrai.factory import BaseFactory


class BaseScheduler(LRScheduler, ABC):
    """Base scheduler class for training schedules."""

    _required_state_keys: Tuple[str, ...] = ()

    def load_state_dict(self, state_dict):
        """Validate schedule settings before restoring PyTorch's complete state."""
        for key in self._required_state_keys:
            if key not in state_dict:
                raise KeyError(key)
        super().load_state_dict(state_dict)

    @abstractmethod
    def get_lr(self) -> List[float]:
        """Calculate the current learning rate."""
        raise NotImplementedError


class SchedulerFactory(BaseFactory[BaseScheduler]):
    """Factory for registered training schedulers."""
