"""Composable sampling strategies for logit transformation.

Implements the Strategy pattern: each sampling technique
(temperature, top-k, top-p) is a pluggable strategy that
can be composed into a pipeline.
"""

from abc import ABC, abstractmethod
from typing import List

import torch
from torch import Tensor


class BaseSamplingStrategy(ABC):
    """Abstract base for a logit transformation strategy."""

    @abstractmethod
    def apply(self, logits: Tensor, filter_value: float = -float("inf")) -> Tensor:
        """Applies the strategy to logits.

        Args:
            logits: Raw logits tensor (batch, vocab_size).
            filter_value: Value assigned to filtered-out positions.

        Returns:
            Transformed logits tensor (may be the same or a new tensor).
        """


class TemperatureStrategy(BaseSamplingStrategy):
    """Divides logits by temperature to control randomness."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def apply(self, logits, filter_value=-float("inf")):
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return logits


class TopKStrategy(BaseSamplingStrategy):
    """Keeps only the top-k logits, setting the rest to filter_value."""

    def __init__(self, top_k: int = 0):
        self.top_k = top_k

    def apply(self, logits, filter_value=-float("inf")):
        if self.top_k > 0:
            k = min(self.top_k, logits.size(-1))
            topk_vals = torch.topk(logits, k, dim=-1)[0]
            threshold = topk_vals[..., -1, None]
            indices = logits < threshold
            logits[indices] = filter_value
        return logits


class TopPStrategy(BaseSamplingStrategy):
    """Nucleus (top-p) filtering: keeps the smallest set of tokens whose
    cumulative probability exceeds top_p."""

    def __init__(self, top_p: float = 1.0):
        self.top_p = top_p

    def apply(self, logits, filter_value=-float("inf")):
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cum_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        return logits


class SamplingPipeline(BaseSamplingStrategy):
    """Composes multiple sampling strategies into a single transformation.

    Strategies are applied sequentially in the order they are provided,
    matching the original temperature → top-k → top-p ordering.
    """

    def __init__(self, strategies: List[BaseSamplingStrategy]):
        self.strategies = strategies

    def apply(self, logits, filter_value=-float("inf")):
        logits = logits.clone()
        for strategy in self.strategies:
            logits = strategy.apply(logits, filter_value)
        return logits


def apply_sampling_strategies(
    logits: Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    filter_value: float = -float("inf"),
) -> Tensor:
    """Applies temperature scaling, top-k filtering, and top-p (nucleus) filtering.

    Backward-compatible function that delegates to the Strategy pattern
    pipeline with TemperatureStrategy → TopKStrategy → TopPStrategy ordering.

    Args:
        logits: Raw logits tensor of shape (batch, vocab_size).
        temperature: Temperature scaling factor (1.0 = no scaling).
        top_k: Keep only top-k logits (0 disables).
        top_p: Nucleus probability threshold (1.0 disables).
        filter_value: Value to assign to filtered-out positions.

    Returns:
        Modified logits tensor with same shape as input.
    """
    pipeline = SamplingPipeline(
        [
            TemperatureStrategy(temperature),
            TopKStrategy(top_k),
            TopPStrategy(top_p),
        ]
    )
    return pipeline.apply(logits, filter_value)
