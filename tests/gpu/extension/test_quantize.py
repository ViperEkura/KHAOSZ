"""Explicit quantize outputs, fused scale publication and boundary checks."""

import itertools

import pytest
import torch

from astrai.extension.kernel.quantize import quantize, quantize_scaled
from astrai.extension.runtime.loader import get_module
from tests.support.capabilities import skip_no_fp8

F4 = torch.float8_e4m3fn
F5 = torch.float8_e5m2
OUTPUTS = [(F4, None), (F5, None), (None, F4), (None, F5)]
OUTPUTS += list(itertools.product((F4, F5), repeat=2))


def _ring(n=4):
    return torch.zeros(n + 4 + get_module("quantize").K_FOLD_SLOTS + 2, device="cuda")


@skip_no_fp8
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dtype,tdtype", OUTPUTS)
@pytest.mark.parametrize("shape", [(37, 67), (2, 5, 65)])
def test_requested_outputs_match_cast(input_dtype, dtype, tdtype, shape):
    torch.manual_seed(17)
    # Include a noncontiguous last axis and flattenable leading dimensions.
    x = torch.randn(*shape, device="cuda", dtype=input_dtype).transpose(-1, -2)
    inv_scale = torch.tensor([0.25], device="cuda")
    out, out_t, amax = quantize(x, inv_scale, dtype=dtype, tdtype=tdtype)
    ref = x.float() * inv_scale
    assert amax is None
    if dtype is None:
        assert out is None
    else:
        assert out.shape == x.shape
        assert torch.equal(out.view(torch.uint8), ref.to(dtype).view(torch.uint8))
    if tdtype is None:
        assert out_t is None
    else:
        expected = ref.reshape(-1, x.shape[-1]).t().contiguous().to(tdtype)
        assert torch.equal(out_t.view(torch.uint8), expected.view(torch.uint8))


@skip_no_fp8
@pytest.mark.parametrize("external", [False, True])
def test_publish_override_and_next_round_multiplier(external):
    ring = _ring()
    ring[:4].fill_(1)
    inv_scale = torch.tensor([2.0], device="cuda")
    next_scale = torch.zeros(1, device="cuda") if external else ring[-2:-1]
    next_inv_scale = torch.zeros(1, device="cuda") if external else ring[-1:]
    x = torch.full((37, 67), 112.0, device="cuda", dtype=torch.bfloat16)
    out, out_t, amax = quantize_scaled(
        x,
        inv_scale,
        ring,
        hist_len=4,
        hist_idx=2,
        dtype=F4,
        tdtype=F5,
        next_scale=next_scale,
        next_inv_scale=next_inv_scale,
    )
    assert amax.data_ptr() == ring[6:7].data_ptr()
    assert float(amax) == 112
    assert float(next_scale) == 0.25
    assert float(next_inv_scale) == 4
    # Publication must leave pair zero and the caller's current multiplier intact.
    assert float(ring[4]) == 0 and float(ring[5]) == 0
    assert float(inv_scale) == 2
    if external:
        assert torch.equal(ring[-2:], torch.zeros(2, device="cuda"))
    torch.testing.assert_close(
        out.float(), torch.full_like(x.float(), 224), rtol=0, atol=0
    )
    assert out_t.shape == (67, 37)
    next_out, _, _ = quantize(x, next_inv_scale, dtype=F4)
    torch.testing.assert_close(
        next_out.float(), torch.full_like(x.float(), 448), rtol=0, atol=0
    )


@skip_no_fp8
def test_transpose_only_default_range_and_margin():
    ring = _ring()
    x = torch.full((2, 3, 7), 112.0, device="cuda", dtype=torch.float16)
    out, out_t, amax = quantize_scaled(
        x,
        torch.ones(1, device="cuda"),
        ring,
        hist_len=4,
        hist_idx=1,
        tdtype=F5,
        margin=1,
    )
    assert out is None and out_t.shape == (7, 6)
    assert float(amax) == 112
    expected = torch.tensor([112.0 / 57344 / 2], device="cuda")
    torch.testing.assert_close(ring[4:5], expected, rtol=0, atol=0)
    torch.testing.assert_close(ring[5:6], expected.reciprocal(), rtol=0, atol=0)


@skip_no_fp8
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "dtype or tdtype"),
        ({"dtype": torch.float32}, "output dtype"),
        ({"tdtype": torch.int8}, "output dtype"),
    ],
)
def test_invalid_output_requests(kwargs, match):
    x = torch.ones(2, 3, device="cuda")
    with pytest.raises(RuntimeError, match=match):
        quantize(x, torch.ones(1, device="cuda"), **kwargs)


@skip_no_fp8
@pytest.mark.parametrize(
    "case,match",
    [
        ("input_dtype", "no kernel"),
        ("scalar_input", "1D"),
        ("transpose_1d", "2D"),
        ("inv_scale_dtype", "inv_scale"),
        ("inv_scale_size", "inv_scale"),
        ("inv_scale_cpu", "inv_scale"),
        ("ring_strided", "contiguous"),
        ("ring_cpu", "ring_state"),
        ("ring_small", "too small"),
        ("hist_zero", "hist_len"),
        ("hist_large", "hist_len"),
        ("hist_wraparound", "hist_len"),
        ("idx_negative", "hist_idx"),
        ("idx_large", "hist_idx"),
        ("idx_wraparound", "hist_idx"),
        ("next_scale_only", "provided together"),
        ("next_inv_scale_only", "provided together"),
        ("next_cpu", "next_scale"),
        ("max_zero", "fp8_max"),
        ("max_nan", "fp8_max"),
        ("margin_large", "margin"),
    ],
)
def test_invalid_scaled_parameters(case, match):
    x = torch.ones(2, 3, device="cuda")
    inv_scale = torch.ones(1, device="cuda")
    ring = _ring()
    args = dict(hist_len=4, hist_idx=0, dtype=F4)
    if case == "input_dtype":
        x = x.double()
    elif case == "scalar_input":
        x = x[0, 0]
    elif case == "transpose_1d":
        x = x[0]
        args = dict(hist_len=4, hist_idx=0, tdtype=F4)
    elif case == "inv_scale_dtype":
        inv_scale = inv_scale.double()
    elif case == "inv_scale_size":
        inv_scale = torch.ones(2, device="cuda")
    elif case == "inv_scale_cpu":
        inv_scale = inv_scale.cpu()
    elif case == "ring_strided":
        ring = torch.zeros(ring.numel() * 2, device="cuda")[::2]
    elif case == "ring_cpu":
        ring = ring.cpu()
    elif case == "ring_small":
        ring = ring[:4]
    elif case == "hist_zero":
        args["hist_len"] = 0
    elif case == "hist_large":
        args["hist_len"] = 2**31
    elif case == "hist_wraparound":
        args["hist_len"] = 2**32 + 4
    elif case == "idx_negative":
        args["hist_idx"] = -1
    elif case == "idx_large":
        args["hist_idx"] = 4
    elif case == "idx_wraparound":
        args["hist_idx"] = 2**32
    elif case == "next_scale_only":
        args["next_scale"] = torch.ones(1, device="cuda")
    elif case == "next_inv_scale_only":
        args["next_inv_scale"] = torch.ones(1, device="cuda")
    elif case == "next_cpu":
        args.update(next_scale=torch.ones(1), next_inv_scale=torch.ones(1))
    elif case == "max_zero":
        args["fp8_max"] = 0
    elif case == "max_nan":
        args["fp8_max"] = float("nan")
    elif case == "margin_large":
        args["margin"] = 1024
    with pytest.raises(RuntimeError, match=match):
        quantize_scaled(x, inv_scale, ring, **args)


@skip_no_fp8
def test_other_device_parameters_rejected():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    x = torch.ones(2, 3, device="cuda:0")
    inv_scale = torch.ones(1, device="cuda:0")
    with pytest.raises(RuntimeError, match="ring_state"):
        quantize_scaled(
            x, inv_scale, _ring().to("cuda:1"), hist_len=4, hist_idx=0, dtype=F4
        )
    with pytest.raises(RuntimeError, match="next_scale"):
        quantize_scaled(
            x,
            inv_scale,
            _ring(),
            hist_len=4,
            hist_idx=0,
            dtype=F4,
            next_scale=torch.ones(1, device="cuda:1"),
            next_inv_scale=torch.ones(1, device="cuda:1"),
        )


@skip_no_fp8
def test_nondefault_stream_and_graph_replay():
    stream = torch.cuda.Stream()
    x = torch.ones(64, 128, device="cuda", dtype=torch.bfloat16)
    inv_scale = torch.ones(1, device="cuda")
    ring = _ring()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            quantize_scaled(
                x, inv_scale, ring, hist_len=4, hist_idx=0, dtype=F4, tdtype=F5
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out, out_t, amax = quantize_scaled(
                x,
                inv_scale,
                ring,
                hist_len=4,
                hist_idx=0,
                dtype=F4,
                tdtype=F5,
            )
        x.fill_(2)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert float(amax) == 2
    assert torch.equal(out.float(), x.float())
    assert torch.equal(out_t.float(), x.t().float())


@skip_no_fp8
@pytest.mark.parametrize("hist_len", [1, 65])
@pytest.mark.parametrize("pairs", [1, 2])
def test_ring_offsets_and_capacity_boundaries(hist_len, pairs):
    scratch_slots = get_module("quantize").K_FOLD_SLOTS
    size = hist_len + 4 + scratch_slots + 2 * (pairs - 1)
    storage = torch.full((size + 2,), -17.0, device="cuda")
    ring = storage[1:-1]
    ring.zero_()
    ring[:hist_len].fill_(7)
    inv_scale = torch.ones(1, device="cuda")
    x = torch.full((37, 67), 14.0, device="cuda")
    _, _, amax = quantize_scaled(
        x,
        inv_scale,
        ring,
        hist_len=hist_len,
        hist_idx=hist_len - 1,
        dtype=F4,
    )
    assert amax.data_ptr() == ring[hist_len + 2 : hist_len + 3].data_ptr()
    assert float(amax) == 14
    expected_history = torch.full((hist_len,), 7.0, device="cuda")
    expected_history[-1] = 14
    torch.testing.assert_close(ring[:hist_len], expected_history, rtol=0, atol=0)
    torch.testing.assert_close(
        ring[hist_len : hist_len + 1],
        torch.tensor([14.0 / 448], device="cuda"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ring[hist_len + 1 : hist_len + 2],
        ring[hist_len : hist_len + 1].reciprocal(),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(ring[hist_len + 3 :]) == 0
    assert float(storage[0]) == float(storage[-1]) == -17
    # One float short must fail before the kernel can write outside this view.
    with pytest.raises(RuntimeError, match="too small"):
        quantize_scaled(
            x,
            inv_scale,
            ring[: hist_len + 3 + scratch_slots],
            hist_len=hist_len,
            hist_idx=hist_len - 1,
            dtype=F4,
        )


@skip_no_fp8
@pytest.mark.parametrize(
    "shape,match",
    [
        ((65535 * 32 + 1, 1), "CUDA grid limit"),
        ((2**31, 1), "too large"),
        ((1, 2**31), "too large"),
    ],
)
def test_invalid_extent_rejected_before_contiguous_copy(shape, match):
    x = torch.ones(1, 1, device="cuda").expand(shape)
    with pytest.raises(RuntimeError, match=match):
        quantize(x, torch.ones(1, device="cuda"), dtype=F4)
