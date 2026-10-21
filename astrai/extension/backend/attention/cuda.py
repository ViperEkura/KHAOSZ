"""Native CUDA attention backend for packed KV-cache inference."""

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from astrai.extension.backend.attention.common import AttentionSpec, _check_fwd
from astrai.extension.kernel.attention import attn_paged_decode, attn_paged_prefill
from astrai.extension.runtime.loader import is_available

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache


def _validate_paged_mask(
    q: Tensor, kv_cache: "KVCache", mask: Tensor, fwd: Optional[str]
) -> None:
    """Check host-visible mask metadata before writing the KV cache."""
    if mask.ndim not in (2, 3, 4):
        raise ValueError("paged mask must be 2D, 3D or 4D")
    if mask.device != q.device or mask.dtype != torch.bool:
        raise ValueError("paged mask must be boolean on Q's device")
    if mask.stride(-1) != 1:
        raise ValueError("paged mask key axis must be contiguous")
    if mask.size(0) not in (1, kv_cache.req_pool_indices.numel()):
        raise ValueError("paged mask batch mismatch")
    if mask.ndim == 4 and mask.size(1) not in (1, q.size(1)):
        raise ValueError("paged mask head mismatch")
    if not 0 < mask.size(-1) <= kv_cache.req_to_token.size(1):
        raise ValueError("paged mask key axis exceeds cache capacity")
    if mask.ndim >= 3:
        max_rows = 1 if fwd == "decode" else q.size(0)
        if not 0 < mask.size(-2) <= max_rows:
            raise ValueError("paged mask query axis must use request-local rows")


HEAD_DIMS = (32, 64, 128, 256)


def supports(spec: AttentionSpec) -> bool:
    return (
        spec.device.type == "cuda"
        and not spec.needs_backward
        and spec.fwd in ("prefill", "decode")
        and spec.kv_cache_kind != "none"
        and (spec.scale is None or spec.scale > 0)
        and (
            spec.mask_kind == "none"
            or (
                spec.mask_kind == str(torch.bool)
                and 2 <= spec.mask_ndim <= 4
                and spec.mask_last_stride == 1
            )
        )
        and spec.ndim == 3
        and spec.dtype in (torch.bfloat16, torch.float16)
        and spec.head_dim in HEAD_DIMS
        and is_available("attention")
    )


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
    if kv_cache is None:
        raise RuntimeError("CUDA attention does not support training (kv_cache=None)")
    if attn_mask is not None:
        _validate_paged_mask(q, kv_cache, attn_mask, fwd)
    if fwd == "decode":
        return _decode(q, k, v, kv_cache, layer_id, attn_mask, is_causal, scale)
    return _prefill(q, k, v, kv_cache, layer_id, attn_mask, is_causal, scale)


def _decode(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: "KVCache",
    layer_id: int,
    attn_mask: Optional[Tensor],
    is_causal: bool,
    scale: Optional[float],
) -> Tensor:
    kv_indptr = kv_cache.kv_indptr

    out = attn_paged_decode(
        q,
        kv_cache.k_buffer[layer_id],
        kv_cache.v_buffer[layer_id],
        kv_cache.req_to_token,
        kv_cache.req_pool_indices,
        kv_indptr,
        new_k=k,
        new_v=v,
        mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        o_part_buf=kv_cache.decode_o_part,
        ml_part_buf=kv_cache.decode_ml_part,
        out_buf=kv_cache.decode_out,
    )
    return out


def _prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: "KVCache",
    layer_id: int,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    loc = kv_cache.out_cache_loc
    kv_cache.k_buffer[layer_id, loc] = k
    kv_cache.v_buffer[layer_id, loc] = v

    out = attn_paged_prefill(
        q,
        kv_cache.k_buffer[layer_id],
        kv_cache.v_buffer[layer_id],
        kv_cache.req_to_token,
        kv_cache.req_pool_indices,
        kv_cache.kv_indptr,
        kv_cache.qo_indptr,
        kv_cache.q_tile_to_batch,
        kv_cache.q_tile_to_index,
        attn_mask,
        is_causal=is_causal,
        scale=scale,
    )
    return out
