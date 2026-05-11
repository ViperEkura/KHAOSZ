"""Inference module for continuous batching.

Layers:
  - engine.py:    Facade (InferenceEngine), Value Object (GenerationParams, GenerationRequest)
  - scheduler.py: Continuous-batching loop, Task state machine, TaskStatus enum
  - cache.py:     PagedCache (page-table-indirected KV cache with alloc/free)
  - sampling.py:  Strategy pattern (TemperatureStrategy, TopKStrategy, TopPStrategy)
  - server.py:    FastAPI HTTP server (OpenAI-compatible endpoints)
"""

from astrai.inference.engine import (
    GenerationParams,
    GenerationRequest,
    InferenceEngine,
)
from astrai.inference.sample import (
    BaseSamplingStrategy,
    SamplingPipeline,
    TemperatureStrategy,
    TopKStrategy,
    TopPStrategy,
    sample,
)
from astrai.inference.scheduler import InferenceScheduler
from astrai.inference.task import STOP, Task, TaskStatus

__all__ = [
    # Engine / Requests
    "InferenceEngine",
    "GenerationRequest",
    "GenerationParams",
    # Scheduler
    "InferenceScheduler",
    "STOP",
    "Task",
    "TaskStatus",
    # Sampling (Strategy pattern)
    "sample",
    "BaseSamplingStrategy",
    "TemperatureStrategy",
    "TopKStrategy",
    "TopPStrategy",
    "SamplingPipeline",
]
