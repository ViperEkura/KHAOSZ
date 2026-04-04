"""
AstrAI Inference Module

Provides inference components for text generation with continuous batching support.

Main Components:
- InferenceEngine: Unified inference engine for continuous batching
- InferenceScheduler: Task scheduling with dynamic batch composition
- Task, TaskStatus: Task management for continuous batching
- GenerationRequest: Request parameters for generation
- apply_sampling_strategies: Sampling utilities for text generation

Author: AstrAI Team
"""

from astrai.inference.engine import (
    GenerationRequest,
    InferenceEngine,
    InferenceScheduler,
    Task,
    TaskStatus,
    apply_sampling_strategies,
)

__all__ = [
    # Engine
    "InferenceEngine",
    "InferenceScheduler",
    "Task",
    "TaskStatus",
    "GenerationRequest",
    # Sampling
    "apply_sampling_strategies",
]
