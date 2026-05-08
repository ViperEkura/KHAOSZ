"""Page-based KV cache with page-table-indirected read/write.

Provides:
  - PagedCache: paged KV cache combining page pool and tensor storage.
"""

from typing import List, Tuple

import torch
from torch import Tensor

STOP = object()


class PagedCache:
    """Paged KV cache with page-table-indirected read/write.

    Combines:
      - Page pool (ref-counted alloc/free via bitmask)
      - KV tensor storage (k_cache, v_cache)

    Call :meth:`bind` to obtain a batch view for the attention layers.
    """

    def __init__(
        self,
        n_layers: int,
        n_pages: int,
        page_size: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.page_size = page_size
        self._free_mask = (1 << n_pages) - 1
        self._refs: List[int] = [0] * n_pages
        self.k_cache = torch.empty(
            (n_layers, n_pages, page_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.empty(
            (n_layers, n_pages, page_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

    def alloc(self) -> int:
        lsb = self._free_mask & -self._free_mask
        if lsb == 0:
            return -1
        idx = lsb.bit_length() - 1
        self._free_mask ^= lsb
        self._refs[idx] = 1
        return idx

    def alloc_n(self, n: int) -> List[int]:
        pages = [self.alloc() for _ in range(n)]
        if any(p < 0 for p in pages):
            for p in pages:
                if p >= 0:
                    self.free(p)
            return []
        return pages

    def free(self, idx: int) -> None:
        self._refs[idx] -= 1
        if self._refs[idx] == 0:
            self._free_mask |= 1 << idx

    def bind(self, page_table: Tensor, total_len: int = 0) -> "CacheView":
        return CacheView(self, page_table, total_len)

    def write(
        self, layer_id: int, page_table: Tensor, start_pos: int, k: Tensor, v: Tensor
    ) -> None:
        seq_len = k.size(1)
        if seq_len == 0:
            return
        page_size = self.page_size
        written = 0
        first_page = start_pos // page_size
        last_page = (start_pos + seq_len - 1) // page_size
        for pi in range(first_page, last_page + 1):
            phys_pages = page_table[:, pi]
            page_start = pi * page_size
            write_start = max(page_start, start_pos)
            write_end = min(page_start + page_size, start_pos + seq_len)
            offset = write_start - page_start
            chunk = write_end - write_start
            self.k_cache[layer_id, phys_pages, offset : offset + chunk] = k[
                :, written : written + chunk
            ]
            self.v_cache[layer_id, phys_pages, offset : offset + chunk] = v[
                :, written : written + chunk
            ]
            written += chunk

    def gather(self, layer_id: int, page_table: Tensor) -> Tuple[Tensor, Tensor]:
        k_parts, v_parts = [], []
        for pi in range(page_table.size(1)):
            phys_pages = page_table[:, pi]
            if not (phys_pages >= 0).any():
                break
            k_parts.append(self.k_cache[layer_id, phys_pages])
            v_parts.append(self.v_cache[layer_id, phys_pages])
        k = torch.cat(k_parts, dim=1)
        v = torch.cat(v_parts, dim=1)
        return k, v


class CacheView:
    """Per-batch view that bundles PagedCache + page_table + total_len.

    Attention layers receive this as ``paged_cache`` and only see
    ``write()`` / ``gather()``, never raw page tables or length params.
    """

    __slots__ = ("_cache", "_page_table", "_total_len")

    def __init__(self, cache: PagedCache, page_table: Tensor, total_len: int = 0):
        self._cache = cache
        self._page_table = page_table
        self._total_len = total_len

    def write(self, layer_id: int, start_pos: int, k: Tensor, v: Tensor) -> None:
        self._cache.write(layer_id, self._page_table, start_pos, k, v)

    def gather(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        k, v = self._cache.gather(layer_id, self._page_table)
        if self._total_len:
            k = k[:, : self._total_len]
            v = v[:, : self._total_len]
        return k, v
