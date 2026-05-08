"""Inference module for continuous batching.

Layers:
  - engine.py:    Facade (InferenceEngine), Value Object (GenerationParams, GenerationRequest)
  - scheduler.py: Continuous-batching loop, Task state machine, TaskStatus enum
  - cache.py:     Object Pool (SlotAllocator), PrefixCacheManager
  - sampling.py:  Strategy pattern (TemperatureStrategy, TopKStrategy, TopPStrategy)
  - server.py:    FastAPI HTTP server (OpenAI-compatible endpoints)
"""

from astrai.inference.engine import (
    GenerationParams,
    GenerationRequest,
    InferenceEngine,
)
from astrai.inference.sampling import (
    BaseSamplingStrategy,
    SamplingPipeline,
    TemperatureStrategy,
    TopKStrategy,
    TopPStrategy,
    apply_sampling_strategies,
)
from astrai.inference.scheduler import (
    InferenceScheduler,
    Task,
    TaskStatus,
)

__all__ = [
    # Engine / Requests
    "InferenceEngine",
    "GenerationRequest",
    "GenerationParams",
    # Scheduler
    "InferenceScheduler",
    "Task",
    "TaskStatus",
    # Sampling (Strategy pattern)
    "apply_sampling_strategies",
    "BaseSamplingStrategy",
    "TemperatureStrategy",
    "TopKStrategy",
    "TopPStrategy",
    "SamplingPipeline",
]
