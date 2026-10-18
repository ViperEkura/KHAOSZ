"""Training executors: gradient accumulation plus the parallel wrapper.

Importing this package registers every executor (``none``, ``ddp``, ``fsdp``)
with :class:`ExecutorFactory`.
"""

from astrai.parallel.executor.accumulate import (
    AccumOptimizer,
    AccumScheduler,
    GradientState,
)
from astrai.parallel.executor.base import (
    BaseExecutor,
    ExecutorFactory,
    NoneExecutor,
    RolloutCapabilities,
    broadcast_state_dict,
    strip_compile_prefix,
)
from astrai.parallel.executor.ddp import DDPExecutor
from astrai.parallel.executor.fsdp import FSDPExecutor

__all__ = [
    "AccumOptimizer",
    "AccumScheduler",
    "BaseExecutor",
    "DDPExecutor",
    "ExecutorFactory",
    "FSDPExecutor",
    "GradientState",
    "NoneExecutor",
    "RolloutCapabilities",
    "broadcast_state_dict",
    "strip_compile_prefix",
]
