"""Page-based KV cache with page-table-indirected read/write.

Provides:
  - PagedCache: paged KV cache combining page pool and tensor storage.
"""

from typing import Dict, List, Tuple

import torch
from torch import Tensor

STOP = object()


def page_hash(token_ids: List[int], page_idx: int, page_size: int) -> int:
    start = page_idx * page_size
    end = min(start + page_size, len(token_ids))
    h = 0
    for i in range(start, end):
        h = (h * 31 + token_ids[i]) & 0xFFFFFFFFFFFFFFFF
    return h


class PagedCache:
    """Paged KV cache with page-table-indirected read/write and persistent prefix caching.

    Combines:
      - Page pool (ref-counted alloc/free via bitmask)
      - KV tensor storage (k_cache, v_cache)
      - Prefix-cache hash lookup (page_content_hash -> physical_page_idx)
      - LRU eviction for persistent cross-batch prefix caching

    Pages with recorded hashes persist after refcount reaches 0 (pinned).
    They are evicted via LRU only when alloc() finds no free pages.
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
        self._page_to_hash: Dict[int, int] = {}
        self._hash_to_page: Dict[int, int] = {}
        self._lru: List[int] = []
        self._pin: List[bool] = [False] * n_pages

    def _touch(self, idx: int) -> None:
        if self._refs[idx] == 0 and idx in self._lru:
            self._lru.remove(idx)
            self._lru.append(idx)

    def _evict_one(self) -> int:
        while self._lru:
            idx = self._lru.pop(0)
            h = self._page_to_hash.pop(idx, None)
            if h is not None:
                self._hash_to_page.pop(h, None)
            self._pin[idx] = False
            self._refs[idx] = 1
            return idx
        return -1

    def record_page(
        self, page_idx: int, token_ids: List[int], logical_page_idx: int
    ) -> None:
        h = page_hash(token_ids, logical_page_idx, self.page_size)
        old_h = self._page_to_hash.pop(page_idx, None)
        if old_h is not None:
            self._hash_to_page.pop(old_h, None)
        self._page_to_hash[page_idx] = h
        self._hash_to_page[h] = page_idx
        self._pin[page_idx] = True
        if page_idx in self._lru:
            self._lru.remove(page_idx)

    def lookup_prefix(self, token_ids: List[int]) -> List[int]:
        full_pages = len(token_ids) // self.page_size
        hits: List[int] = []
        for i in range(full_pages):
            h = page_hash(token_ids, i, self.page_size)
            p = self._hash_to_page.get(h)
            if p is None:
                break
            self._touch(p)
            hits.append(p)
        return hits

    def inc_ref(self, idx: int) -> None:
        self._refs[idx] += 1
        if self._refs[idx] == 1 and idx in self._lru:
            self._lru.remove(idx)

    def alloc(self) -> int:
        if self._free_mask:
            lsb = self._free_mask & -self._free_mask
            idx = lsb.bit_length() - 1
            self._free_mask ^= lsb
            self._refs[idx] = 1
            if idx in self._lru:
                self._lru.remove(idx)
            return idx
        return self._evict_one()

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
            h = self._page_to_hash.get(idx)
            if h is not None and self._pin[idx]:
                self._lru.append(idx)
            else:
                self._free_mask |= 1 << idx
                h = self._page_to_hash.pop(idx, None)
                if h is not None:
                    self._hash_to_page.pop(h, None)
                self._pin[idx] = False

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
        # page_table: [batch, max_pages] with -1 padding for tasks with fewer pages.
        # clamp(min=0) maps -1 to page 0 (irrelevant data) — truncated by CacheView total_len.
        safe = page_table.clamp(min=0)
        k = self.k_cache[layer_id, safe]
        v = self.v_cache[layer_id, safe]
        k = k.flatten(1, 2)
        v = v.flatten(1, 2)
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
