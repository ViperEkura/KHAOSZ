"""Training learning-rate schedulers and factory registration."""

from astrai.optimization.scheduler.base import BaseScheduler, SchedulerFactory
from astrai.optimization.scheduler.cosine import CosineScheduler
from astrai.optimization.scheduler.sgdr import SGDRScheduler
from astrai.optimization.scheduler.wsd import WSDScheduler

__all__ = [
    "BaseScheduler",
    "CosineScheduler",
    "SGDRScheduler",
    "SchedulerFactory",
    "WSDScheduler",
]
