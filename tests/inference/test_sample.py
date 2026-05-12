"""Unit tests for inference sampling strategies."""

import torch

from astrai.inference.sample import (
    SamplingPipeline,
    TemperatureStrategy,
    TopKStrategy,
    TopPStrategy,
    sample,
)


def test_temperature_scalar():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    s = TemperatureStrategy(0.5)
    result = s.apply(logits.clone())
    assert torch.allclose(result, logits / 0.5)


def test_temperature_skip_when_one():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    s = TemperatureStrategy(1.0)
    result = s.apply(logits.clone())
    assert torch.equal(result, logits)


def test_temperature_per_sample_tensor():
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    s = TemperatureStrategy(torch.tensor([0.5, 0.5]))
    result = s.apply(logits.clone())
    assert torch.allclose(result, logits / 0.5)


def test_top_k_keeps_top():
    logits = torch.tensor([[0.1, 0.5, 0.3, 0.9, 0.2]])
    s = TopKStrategy(top_k=2)
    result = s.apply(logits.clone(), filter_value=-1e9)
    kept = (result > -1e9).sum().item()
    assert kept == 2


def test_top_k_skip_when_zero():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    s = TopKStrategy(top_k=0)
    result = s.apply(logits.clone())
    assert torch.equal(result, logits)


def test_top_k_batch_tensor():
    """When top_k is a batch tensor, max element governs k for all rows."""
    logits = torch.tensor([[0.1, 0.5, 0.3], [0.9, 0.2, 0.1]])
    s = TopKStrategy(top_k=torch.tensor([2, 1]))
    result = s.apply(logits.clone(), filter_value=-1e9)
    assert (result[0] > -1e9).sum() == 2
    assert (result[1] > -1e9).sum() == 2


def test_top_p_nucleus_filtering():
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
    s = TopPStrategy(top_p=0.5)
    result = s.apply(logits.clone(), filter_value=-1e9)
    kept = (result > -1e9).sum().item()
    assert kept >= 1


def test_top_p_skip_when_one():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    s = TopPStrategy(top_p=1.0)
    result = s.apply(logits.clone())
    assert torch.equal(result, logits)


def test_top_p_filter_all_except_max_when_zero():
    logits = torch.tensor([[0.1, 0.5, 0.3, 0.9, 0.2]])
    s = TopPStrategy(top_p=0.0)
    result = s.apply(logits.clone(), filter_value=-1e9)
    kept = (result > -1e9).sum().item()
    assert kept == 1


def test_sampling_pipeline_composes_strategies():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pipeline = SamplingPipeline(
        [
            TemperatureStrategy(0.8),
            TopKStrategy(3),
            TopPStrategy(0.95),
        ]
    )
    result = pipeline.apply(logits.clone(), filter_value=-1e9)
    kept = (result > -1e9).sum().item()
    assert 1 <= kept <= 3


def test_sampling_pipeline_sample_returns_valid_token():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pipeline = SamplingPipeline(
        [
            TemperatureStrategy(0.8),
            TopKStrategy(3),
            TopPStrategy(0.95),
        ]
    )
    tokens = pipeline.sample(logits)
    assert tokens.shape == (1,)
    assert 0 <= tokens[0] < logits.size(-1)


def test_module_sample_shortcut():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    tokens = sample(logits, temperature=0.8, top_k=3, top_p=0.95)
    assert tokens.shape == (1,)
    assert 0 <= tokens[0] < logits.size(-1)


def test_module_sample_batch():
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    tokens = sample(logits, temperature=0.8, top_k=3, top_p=0.95)
    assert tokens.shape == (2,)
    for t in tokens:
        assert 0 <= t < logits.size(-1)
