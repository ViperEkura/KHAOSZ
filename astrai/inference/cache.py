from typing import Dict, List, Tuple

import torch
from torch import Tensor


def page_hash(token_ids: List[int], page_idx: int, page_size: int) -> int:
    start = page_idx * page_size
    end = min(start + page_size, len(token_ids))
    h = 0
    for i in range(start, end):
        h = (h * 31 + token_ids[i]) & 0xFFFFFFFFFFFFFFFF
    return h


class PagedCache:
    """Paged KV cache: page pool, prefix-cache lookup, LRU eviction, task-page mapping."""

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
        self._task_pages: Dict[str, List[int]] = {}
        self._task_cached: Dict[str, int] = {}

    def pages_needed(self, n_tokens: int) -> int:
        return (n_tokens + self.page_size - 1) // self.page_size

    def task_alloc(self, task_id: str, prompt_ids: List[int]) -> bool:
        hit_pages = self.lookup_prefix(prompt_ids)
        cached_tokens = len(hit_pages) * self.page_size
        for p in hit_pages:
            self.inc_ref(p)

        remaining = len(prompt_ids) - cached_tokens
        n_new = self.pages_needed(remaining) if remaining > 0 else 0
        new_pages = self.alloc_n(n_new) if n_new > 0 else []

        if remaining > 0 and not new_pages:
            for p in hit_pages:
                self.free(p)
            return False

        page_table = hit_pages + new_pages
        self._task_pages[task_id] = page_table
        self._task_cached[task_id] = cached_tokens
        return True

    def task_free(self, task_id: str) -> None:
        page_table = self._task_pages.pop(task_id, None)
        self._task_cached.pop(task_id, None)
        if page_table:
            for idx in page_table:
                self.free(idx)

    def task_extend(self, task_id: str, pos: int) -> bool:
        needed = self.pages_needed(pos + 1)
        page_table = self._task_pages[task_id]
        while len(page_table) < needed:
            p = self.alloc()
            if p < 0:
                return False
            page_table.append(p)
        return True

    def task_cached(self, task_id: str) -> int:
        return self._task_cached.get(task_id, 0)

    def task_page_table(self, task_id: str) -> List[int]:
        return self._task_pages.get(task_id, [])

    def task_n_pages(self, task_id: str) -> int:
        return len(self._task_pages.get(task_id, []))

    def task_record_hashes(
        self, task_id: str, prompt_ids: List[int], start_logical_page: int = 0
    ) -> None:
        page_table = self._task_pages[task_id]
        full_pages = len(prompt_ids) // self.page_size
        for i in range(start_logical_page, full_pages):
            self.record_page(page_table[i], prompt_ids, i)

    def make_table_tensor(self, task_ids: List[str], device: torch.device) -> Tensor:
        states = [self._task_pages.get(tid, []) for tid in task_ids]
        max_pages = max((len(s) for s in states), default=0)
        rows = [s + [-1] * (max_pages - len(s)) for s in states]
        return torch.tensor(rows, dtype=torch.long, device=device)

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

    def gather(
        self, layer_id: int, page_table: Tensor, total_len: int
    ) -> Tuple[Tensor, Tensor]:
        safe = page_table.clamp(min=0)
        k = self.k_cache[layer_id, safe]
        v = self.v_cache[layer_id, safe]
        k = k.flatten(1, 2)
        v = v.flatten(1, 2)
        k = k[:, :total_len]
        v = v[:, :total_len]
        return k, v


class CacheView:
    """Bundles PagedCache + page_table + total_len for attention layers."""

    __slots__ = ("_cache", "_page_table", "_total_len")

    def __init__(self, cache: PagedCache, page_table: Tensor, total_len: int = 0):
        self._cache = cache
        self._page_table = page_table
        self._total_len = total_len

    def write(self, layer_id: int, start_pos: int, k: Tensor, v: Tensor) -> None:
        self._cache.write(layer_id, self._page_table, start_pos, k, v)

    def gather(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        return self._cache.gather(layer_id, self._page_table, self._total_len)
