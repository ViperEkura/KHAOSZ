"""Inference module for continuous batching.

Layers:
  - core/:           Core inference loop (cache, executor, scheduler, task)
  - api/:            HTTP protocol handlers (OpenAI, Anthropic)
  - engine.py:       Facade (InferenceEngine), Value Object (GenerationParams, GenerationRequest)
  - sample.py:       Strategy pattern (TemperatureStrategy, TopKStrategy, TopPStrategy)
"""

from astrai.inference.api import (
    AnthropicHandler,
    AnthropicMessage,
    ChatCompletionRequest,
    ChatMessage,
    MessagesRequest,
    OpenAIHandler,
    ProtocolHandler,
    StopChecker,
    StreamContext,
    app,
    run_server,
)
from astrai.inference.core import (
    STOP,
    CacheView,
    Executor,
    InferenceScheduler,
    PagedCache,
    PagePool,
    PrefixCache,
    Task,
    TaskManager,
    TaskStatus,
    TaskTable,
    page_hash,
)
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

__all__ = [
    # Engine / Requests
    "InferenceEngine",
    "GenerationRequest",
    "GenerationParams",
    # Core scheduler
    "InferenceScheduler",
    "Executor",
    "STOP",
    "Task",
    "TaskManager",
    "TaskStatus",
    # Core cache
    "CacheView",
    "PagedCache",
    "PagePool",
    "PrefixCache",
    "TaskTable",
    "page_hash",
    # Sampling (Strategy pattern)
    "sample",
    "BaseSamplingStrategy",
    "TemperatureStrategy",
    "TopKStrategy",
    "TopPStrategy",
    "SamplingPipeline",
    # Protocol
    "ProtocolHandler",
    "StopChecker",
    "StreamContext",
    "AnthropicHandler",
    "OpenAIHandler",
    # Server
    "ChatMessage",
    "ChatCompletionRequest",
    "AnthropicMessage",
    "MessagesRequest",
    "app",
    "run_server",
]
