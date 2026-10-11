"""The shared int8 primitive works without a compiled CUDA extension."""

import pytest
import torch

from astrai.extension.kernel.quantize import quantize_int8


@pytest.mark.parametrize("shape", [(7, 19), (2, 3, 19), (2, 3, 4, 19)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_int8_quantizes_each_last_axis_row(shape, dtype):
    torch.manual_seed(91)
    x = torch.randn(*shape, dtype=dtype).transpose(-1, -2).requires_grad_()
    q, scale = quantize_int8(x)
    rows = x.detach().float().reshape(-1, x.shape[-1])
    expected_scale = rows.abs().amax(-1).clamp_min(1e-12) / 127
    expected = (rows / expected_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    assert q.shape == x.shape and q.dtype == torch.int8 and q.is_contiguous()
    assert scale.dtype == torch.float32 and scale.shape == (rows.shape[0],)
    assert not scale.requires_grad
    assert torch.equal(q.reshape(rows.shape), expected)
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


def test_zero_rows_stay_zero_and_have_positive_scales():
    q, scale = quantize_int8(torch.zeros(2, 3, 17))
    assert torch.count_nonzero(q) == 0
    assert scale.shape == (6,)
    torch.testing.assert_close(
        scale, torch.full_like(scale, 1e-12 / 127), rtol=0, atol=0
    )
