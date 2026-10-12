"""Base class for training learning-rate schedulers."""

from abc import ABC, abstractmethod

from torch.optim.lr_scheduler import LRScheduler


class BaseScheduler(LRScheduler, ABC):
    """Base scheduler class for training schedules."""

    def __init__(self, optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)

    @abstractmethod
    def get_lr(self) -> list[float]:
        """Calculate the current learning rate."""
        raise NotImplementedError
