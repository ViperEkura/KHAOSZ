"""Optional FlashAttention strategy."""

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from astrai.extension.backend.attention.common import (
    AttentionSpec,
    _check_fwd,
    _flash_attn,
    flash_attn_available,
)

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache


def supports(spec: AttentionSpec) -> bool:
    if (
        spec.device.type != "cuda"
        or not flash_attn_available()
        or spec.mask_kind != "none"
    ):
        return False
    if spec.dtype not in (torch.float16, torch.bfloat16):
        return False
    if spec.fwd is not None:
        return spec.ndim == 3 and hasattr(_flash_attn, "flash_attn_varlen_func")
    return True


def forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Optional["KVCache"],
    layer_id: int,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    fwd: Optional[str] = None,
    *,
    scale: Optional[float] = None,
) -> Tensor:
    _check_fwd(fwd)
    if attn_mask is not None:
        raise ValueError("Flash attention cannot handle a custom attention mask")
    # Decode is always packed; prefill/training split by layout.
    if fwd == "decode" or q.ndim == 3:
        return _forward_packed(q, k, v, kv_cache, layer_id, is_causal, scale)
    return _forward_dense(q, k, v, attn_mask, is_causal, scale)


def _forward_dense(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    if attn_mask is not None:
        raise ValueError(
            "Flash attention cannot handle a custom attention mask; "
            "use a causal mask or select Torch attention."
        )
    fa = _flash_attn
    if fa is None:
        raise RuntimeError(
            "Flash attention requires the optional 'flash-attn' package. "
            "Install with `pip install flash-attn`."
        )
    out = fa.flash_attn_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        causal=is_causal,
        softmax_scale=scale,
    )
    return out.contiguous()


def _forward_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: "KVCache",
    layer_id: int,
    is_causal: bool,
    scale: Optional[float],
) -> Tensor:
    fa = _flash_attn
    if fa is None or not hasattr(fa, "flash_attn_varlen_func"):
        raise RuntimeError("packed inference requires flash_attn_varlen_func")
    kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
    kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v
    page_table = kv_cache.req_to_token[kv_cache.req_pool_indices, : kv_cache.max_len]
    positions = torch.arange(kv_cache.max_len, device=q.device)
    indices = page_table[positions.unsqueeze(0) < kv_cache.seq_lens.unsqueeze(1)]
    k_flat = kv_cache.k_buffer[layer_id, indices].contiguous()
    v_flat = kv_cache.v_buffer[layer_id, indices].contiguous()
    out = fa.flash_attn_varlen_func(
        q.contiguous(),
        k_flat,
        v_flat,
        kv_cache.qo_indptr,
        kv_cache.kv_indptr,
        int((kv_cache.qo_indptr[1:] - kv_cache.qo_indptr[:-1]).max()),
        int(kv_cache.seq_lens.max()),
        dropout_p=0.0,
        causal=is_causal,
        softmax_scale=scale,
    )
    return out
