"""Explicit scaled casts and fused history updates.

The caller supplies the current multiplier and selects each output dtype.
Scale selection and state advancement belong to the policy/linear layer.
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
    dimensions. Transposed output requires a 2D+ input. Input may be bf16,
    fp16 or fp32; outputs may be E4M3FN, E5M2 or symmetric int8. The float32
    multiplier broadcasts over flattened [rows, cols]: one element is scalar,
    [rows, 1] scales rows, [cols] or [1, cols] scales columns, and
    [rows, cols] scales elements. Noncontiguous and expanded views retain
    their strides. A nonscalar 1D multiplier follows column broadcasting.
    Int8 rounds to nearest even and clamps to [-127, 127]; NaN maps to zero
    and infinities saturate. No amax is measured.
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
    quant_max: Optional[float] = None,
    next_scale: Optional[torch.Tensor] = None,
    next_inv_scale: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """Quantize, fold raw amax into history, and publish the next scale.

    Returns (out, out_t, amax); amax is a view of the ring's persistent slot.
    The scale is max(max(history) / quant_max / 2**margin, 1e-12), with its
    rounded reciprocal published alongside it. The default range follows
    dtype, or tdtype when only the transpose is requested. Mixed outputs
    share the scalar inv_scale; row multipliers are not supported here.
    Both publication tensors must be provided together; omitting them
    selects the ring's first pair. The caller advances its
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
        quant_max=quant_max,
        next_scale=next_scale,
        next_inv_scale=next_inv_scale,
    )


__all__ = ["quantize", "quantize_scaled"]
