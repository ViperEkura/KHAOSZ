"""Unit tests for inference cache components."""

import torch

from astrai.inference.cache import (
    PagedCache,
    PagePool,
    PrefixCache,
    TaskTable,
    page_hash,
)


def test_page_hash_full_page():
    token_ids = list(range(256))
    h = page_hash(token_ids, 0, 64)
    assert isinstance(h, int)
    assert h >= 0


def test_page_hash_different_page_differs():
    token_ids = list(range(256))
    assert page_hash(token_ids, 0, 64) != page_hash(token_ids, 1, 64)


def test_page_pool_alloc_free_cycle():
    pool = PagePool(n_pages=4)
    a = pool.alloc()
    b = pool.alloc()
    assert a != b
    pool.free(a)
    pool.free(b)
    c = pool.alloc()
    assert c in (a, b)


def test_page_pool_alloc_when_full():
    pool = PagePool(n_pages=2)
    pool.alloc()
    pool.alloc()
    assert pool.alloc() == -1


def test_page_pool_lru_eviction():
    evicted = []

    def on_evict(idx):
        evicted.append(idx)

    pool = PagePool(n_pages=2, on_evict=on_evict)
    p0 = pool.alloc()
    p1 = pool.alloc()
    pool.free(p0, keep_cached=True)
    pool.free(p1, keep_cached=True)
    pool.alloc()
    assert len(evicted) == 1
    assert evicted[0] == p0


def test_page_pool_inc_ref_and_free():
    pool = PagePool(n_pages=2)
    p = pool.alloc()
    pool.inc_ref(p)
    assert pool._refs[p] == 2
    pool.free(p)
    assert pool._refs[p] == 1
    pool.free(p)
    assert pool._refs[p] == 0


def test_page_pool_touch_moves_to_end():
    pool = PagePool(n_pages=4)
    p0 = pool.alloc()
    p1 = pool.alloc()
    p2 = pool.alloc()
    pool.free(p0, keep_cached=True)
    pool.free(p1, keep_cached=True)
    pool.free(p2, keep_cached=True)
    assert next(iter(pool._lru)) == p0
    pool.touch(p0)
    assert next(reversed(pool._lru)) == p0


def test_page_pool_remove_from_lru():
    pool = PagePool(n_pages=4)
    p0 = pool.alloc()
    pool.free(p0, keep_cached=True)
    assert p0 in pool._lru
    pool.remove_from_lru(p0)
    assert p0 not in pool._lru


def test_page_pool_keep_cached_realloc():
    """Free mask has priority over LRU; cached page returned only when no free pages."""
    pool = PagePool(n_pages=3)
    p0 = pool.alloc()
    p1 = pool.alloc()
    p2 = pool.alloc()
    pool.free(p0, keep_cached=True)
    pool.free(p1, keep_cached=True)
    pool.free(p2, keep_cached=True)
    assert pool.alloc() == p0


def _record_then_cache(pool, prefix, page, token_ids, logical_idx):
    """Simulate the real lifecycle: record → ref stays >0, then free cached returns to LRU."""
    prefix.record(page, token_ids, logical_idx, pool)
    pool.free(page, keep_cached=True)


def test_prefix_cache_lookup_returns_hits():
    token_ids = list(range(256))
    pool = PagePool(n_pages=16)
    prefix = PrefixCache(page_size=64)
    pages = [pool.alloc() for _ in range(4)]
    for i, p in enumerate(pages):
        _record_then_cache(pool, prefix, p, token_ids, i)
    hits = prefix.lookup(token_ids, pool)
    assert hits == pages


def test_prefix_cache_lookup_stops_at_first_miss():
    token_ids = list(range(256))
    pool = PagePool(n_pages=16)
    prefix = PrefixCache(page_size=64)
    p0 = pool.alloc()
    _record_then_cache(pool, prefix, p0, token_ids, 0)
    p1 = pool.alloc()
    _record_then_cache(pool, prefix, p1, [99] * 64, 1)
    hits = prefix.lookup(token_ids, pool)
    assert len(hits) == 1
    assert hits[0] == p0


def test_prefix_cache_ignores_partial_last_page():
    token_ids = list(range(100))
    pool = PagePool(n_pages=16)
    prefix = PrefixCache(page_size=64)
    p = pool.alloc()
    _record_then_cache(pool, prefix, p, token_ids, 0)
    hits = prefix.lookup(token_ids, pool)
    assert len(hits) == 1


def test_prefix_cache_on_evict_clears_mappings():
    pool = PagePool(n_pages=4)
    prefix = PrefixCache(page_size=64)
    p = pool.alloc()
    _record_then_cache(pool, prefix, p, list(range(64)), 0)
    assert prefix.has_page(p)
    prefix.on_evict(p)
    assert not prefix.has_page(p)


def test_prefix_cache_has_page():
    pool = PagePool(n_pages=4)
    prefix = PrefixCache(page_size=64)
    p = pool.alloc()
    assert not prefix.has_page(p)
    _record_then_cache(pool, prefix, p, list(range(64)), 0)
    assert prefix.has_page(p)


def test_task_table_set_get():
    pool = PagePool(n_pages=8)
    table = TaskTable(pool, page_size=64)
    table.set("task1", [0, 1, 2], 128)
    assert table.get("task1") == [0, 1, 2]
    assert table.get_cached("task1") == 128


def test_task_table_get_missing():
    pool = PagePool(n_pages=8)
    table = TaskTable(pool, page_size=64)
    assert table.get("nonexistent") == []
    assert table.get_cached("nonexistent") == 0


def test_task_table_pop():
    pool = PagePool(n_pages=8)
    table = TaskTable(pool, page_size=64)
    table.set("task1", [0, 1], 64)
    pages, cached = table.pop("task1")
    assert pages == [0, 1]
    assert cached == 64
    assert table.get("task1") == []


def test_task_table_extend_allocates_pages():
    pool = PagePool(n_pages=8)
    table = TaskTable(pool, page_size=64)
    table.set("task1", [], 0)
    ok = table.extend("task1", 200)
    assert ok
    assert len(table.get("task1")) == 4


def test_task_table_extend_fails_when_pool_full():
    pool = PagePool(n_pages=2)
    table = TaskTable(pool, page_size=64)
    table.set("task1", [pool.alloc(), pool.alloc()], 0)
    ok = table.extend("task1", 300)
    assert not ok


def test_task_table_table_tensor():
    pool = PagePool(n_pages=16)
    table = TaskTable(pool, page_size=64)
    table.set("a", [0, 1], 0)
    table.set("b", [2, 3, 4], 0)
    t = table.table_tensor(["a", "b"], torch.device("cpu"))
    assert t.shape == (2, 3)
    assert t[0].tolist() == [0, 1, -1]
    assert t[1].tolist() == [2, 3, 4]


def test_task_table_table_tensor_empty_input():
    pool = PagePool(n_pages=4)
    table = TaskTable(pool, page_size=64)
    t = table.table_tensor([], torch.device("cpu"))
    assert t.numel() == 0


def test_paged_cache_write_gather_single_page():
    cache = PagedCache(
        n_layers=2,
        n_pages=8,
        page_size=4,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    page_table = torch.tensor([[0]], dtype=torch.long)
    k = torch.randn(1, 2, 2, 8)
    v = torch.randn(1, 2, 2, 8)

    cache.write(0, page_table, 0, k, v)
    gk, gv = cache.gather(0, page_table, 2)
    assert torch.allclose(gk, k)


def test_paged_cache_write_cross_page():
    cache = PagedCache(
        n_layers=1,
        n_pages=8,
        page_size=4,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    page_table = torch.tensor([[0, 1]], dtype=torch.long)
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)

    cache.write(0, page_table, 0, k, v)
    gk, gv = cache.gather(0, page_table, 8)
    assert torch.allclose(gk, k)


def test_paged_cache_gather_truncates_to_total_len():
    cache = PagedCache(
        n_layers=1,
        n_pages=8,
        page_size=4,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    page_table = torch.tensor([[0, 1]], dtype=torch.long)
    k = torch.randn(1, 6, 2, 8)
    v = torch.randn(1, 6, 2, 8)
    cache.write(0, page_table, 0, k, v)

    gk, gv = cache.gather(0, page_table, 5)
    assert gk.shape == (1, 5, 2, 8)


def test_paged_cache_gather_clamps_negative_padding():
    cache = PagedCache(
        n_layers=1,
        n_pages=8,
        page_size=4,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    page_table = torch.tensor([[0, -1]], dtype=torch.long)
    gk, gv = cache.gather(0, page_table, 4)
    assert gk.shape == (1, 4, 2, 8)
