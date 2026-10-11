"""Explicit FP8 scaled casts and fused history updates.

The caller supplies the current multiplier and selects each output dtype.
FP8 recipe selection and state advancement belong to the policy/linear layer.
"""

from typing import Optional, Tuple

import torch

from astrai.extension.runtime.loader import get_module


def quantize(
    x: torch.Tensor,
    inv_scale: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    tdtype: Optional[torch.dtype] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
    """Cast ``x * inv_scale``; return ``(out, out_t, None)``.

    Each dtype requests one orientation; at least one is required. ``out``
    preserves x's shape and ``out_t`` is [cols, rows] after flattening leading
    dimensions. Transposed output requires a 2D+ input. No amax is measured.
    """
    return get_module("quantize").quantize(x, inv_scale, dtype=dtype, tdtype=tdtype)


def quantize_scaled(
    x: torch.Tensor,
    inv_scale: torch.Tensor,
    ring_state: torch.Tensor,
    *,
    hist_len: int,
    hist_idx: int,
    dtype: Optional[torch.dtype] = None,
    tdtype: Optional[torch.dtype] = None,
    margin: int = 0,
    fp8_max: Optional[float] = None,
    next_scale: Optional[torch.Tensor] = None,
    next_inv_scale: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Quantize, fold raw amax into history, and publish the next scale.

    Returns (out, out_t, amax); amax is a view of the ring's persistent slot.
    The scale is max(max(history) / fp8_max / 2**margin, 1e-12), with its
    rounded reciprocal published alongside it. The default range follows
    dtype, or tdtype when only the transpose is requested. Mixed outputs
    share inv_scale. Both publication tensors must be provided together;
    omitting them selects the ring's first pair. The caller advances its
    cursor and switches its current pair; this operation does neither.
    """
    return get_module("quantize").quantize_scaled(
        x,
        inv_scale,
        ring_state,
        hist_len=hist_len,
        hist_idx=hist_idx,
        dtype=dtype,
        tdtype=tdtype,
        margin=margin,
        fp8_max=fp8_max,
        next_scale=next_scale,
        next_inv_scale=next_inv_scale,
    )


def quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 quantization of each last-axis row.

    Returns contiguous int8 data with x's shape and a flat float32 scale
    vector. For [N, K] weights the scales are per output channel; for
    [..., K] activations they are per flattened row. Dequantization is
    q.reshape(-1, K).float() * scale[:, None]. Arithmetic is float32 and
    detached from autograd.
    """
    rows = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
    scale = rows.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    q = torch.round(rows / scale.unsqueeze(-1)).clamp_(-127, 127)
    return q.to(torch.int8).reshape(x.shape).contiguous(), scale.contiguous()


__all__ = ["quantize", "quantize_scaled", "quantize_int8"]
