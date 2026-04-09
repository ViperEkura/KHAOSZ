"""Tests for scheduler concurrency."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from astrai.inference.scheduler import (
    InferenceScheduler,
    PrefixCacheManager,
)


def test_prefix_cache_concurrent_insert_find():
    """Test concurrent insert and find operations."""
    cache = PrefixCacheManager(max_capacity=100)

    results = {"errors": [], "inserts": 0, "finds": 0}

    def insert_worker():
        try:
            for i in range(50):
                cache.insert((i,), slot=i % 10)
                results["inserts"] += 1
        except Exception as e:
            results["errors"].append(str(e))

    def find_worker():
        try:
            for i in range(50):
                cache.find_longest_prefix([i])
                results["finds"] += 1
        except Exception as e:
            results["errors"].append(str(e))

    threads = [threading.Thread(target=insert_worker) for _ in range(3)]
    threads += [threading.Thread(target=find_worker) for _ in range(3)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert results["inserts"] == 150
    assert results["finds"] == 150


def test_prefix_cache_concurrent_release():
    """Test concurrent release operations."""
    cache = PrefixCacheManager(max_capacity=100)

    # Insert some prefixes
    for i in range(10):
        cache.insert((i,), slot=i)

    results = {"errors": []}

    def release_worker():
        try:
            for i in range(10):
                cache.release((i,))
        except Exception as e:
            results["errors"].append(str(e))

    threads = [threading.Thread(target=release_worker) for _ in range(3)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"


def test_prefix_cache_concurrent_insert_release_find():
    """Test mixed concurrent operations."""
    cache = PrefixCacheManager(max_capacity=50)

    results = {"errors": []}

    def worker(worker_id):
        try:
            for i in range(20):
                token_ids = (worker_id * 100 + i,)
                cache.insert(token_ids, slot=worker_id)

                # Find after insert
                cache.find_longest_prefix(list(token_ids))

                # Release
                cache.release(token_ids)
        except Exception as e:
            results["errors"].append(f"Worker {worker_id}: {str(e)}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer."""
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.config.n_kv_heads = 8
    mock_model.config.n_heads = 8
    mock_model.config.dim = 128
    mock_model.config.n_layers = 2
    mock_model.config.max_len = 100

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "token"
    mock_tokenizer.stop_ids = [0]
    mock_tokenizer.pad_id = None

    return mock_model, mock_tokenizer


def test_scheduler_concurrent_add_task(mock_model_and_tokenizer):
    """Test concurrent add_task operations."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.scheduler.AutoModel"):
        with patch("astrai.inference.scheduler.AutoTokenizer"):
            scheduler = InferenceScheduler(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_batch_size=4,
                device="cpu",
            )

    results = {"task_ids": [], "errors": []}
    lock = threading.Lock()

    def add_task_worker(worker_id):
        try:
            for i in range(10):
                task_id = scheduler.add_task(f"prompt from worker {worker_id}-{i}")
                with lock:
                    results["task_ids"].append(task_id)
        except Exception as e:
            results["errors"].append(str(e))

    threads = [threading.Thread(target=add_task_worker, args=(i,)) for i in range(5)]

    for t in threads:
        t.start()

    # Let some tasks be processed
    time.sleep(0.1)

    scheduler.stop()

    for t in threads:
        t.join()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["task_ids"]) == 50


def test_scheduler_concurrent_add_remove_task(mock_model_and_tokenizer):
    """Test concurrent add and remove task operations."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.scheduler.AutoModel"):
        with patch("astrai.inference.scheduler.AutoTokenizer"):
            scheduler = InferenceScheduler(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_batch_size=4,
                device="cpu",
            )

    results = {"added": [], "removed": [], "errors": []}

    def add_worker():
        try:
            for i in range(20):
                task_id = scheduler.add_task(f"prompt {i}")
                results["added"].append(task_id)
                time.sleep(0.001)
        except Exception as e:
            results["errors"].append(f"Add: {str(e)}")

    def remove_worker():
        try:
            time.sleep(0.05)  # Wait for some tasks to be added
            for task_id in results["added"][:10]:
                scheduler.remove_task(task_id)
                results["removed"].append(task_id)
        except Exception as e:
            results["errors"].append(f"Remove: {str(e)}")

    add_thread = threading.Thread(target=add_worker)
    remove_thread = threading.Thread(target=remove_worker)

    add_thread.start()
    remove_thread.start()

    time.sleep(0.2)
    scheduler.stop()

    add_thread.join()
    remove_thread.join()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["added"]) == 20


def test_scheduler_concurrent_get_stats(mock_model_and_tokenizer):
    """Test concurrent get_stats operations."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.scheduler.AutoModel"):
        with patch("astrai.inference.scheduler.AutoTokenizer"):
            scheduler = InferenceScheduler(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_batch_size=4,
                device="cpu",
            )

    results = {"stats": [], "errors": []}

    def add_tasks():
        try:
            for i in range(20):
                scheduler.add_task(f"prompt {i}")
                time.sleep(0.001)
        except Exception as e:
            results["errors"].append(f"Add: {str(e)}")

    def get_stats():
        try:
            for _ in range(50):
                stats = scheduler.get_stats()
                results["stats"].append(stats)
                time.sleep(0.001)
        except Exception as e:
            results["errors"].append(f"Get stats: {str(e)}")

    add_thread = threading.Thread(target=add_tasks)
    stats_thread = threading.Thread(target=get_stats)

    add_thread.start()
    stats_thread.start()

    time.sleep(0.3)
    scheduler.stop()

    add_thread.join()
    stats_thread.join()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["stats"]) == 50

    # Verify stats are consistent
    for stats in results["stats"]:
        assert "total_tasks" in stats
        assert stats["total_tasks"] >= 0


def test_prefix_cache_insert_same_prefix_concurrently():
    """Test inserting the same prefix concurrently."""
    cache = PrefixCacheManager(max_capacity=100)

    results = {"slot_values": [], "errors": []}

    def insert_worker():
        try:
            # All workers try to insert the same prefix
            cache.insert((1, 2, 3), slot=threading.current_thread().name)
            node = cache.root.children.get(1)
            if node:
                node = node.children.get(2)
                if node:
                    node = node.children.get(3)
                    if node:
                        results["slot_values"].append(node.slot)
        except Exception as e:
            results["errors"].append(str(e))

    threads = [threading.Thread(target=insert_worker) for _ in range(10)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All inserts should succeed, final slot should be one of the values
    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    # Check ref_count is correct (should be 10)
    node = cache.root.children.get(1).children.get(2).children.get(3)
    assert node.ref_count == 10, f"Expected ref_count=10, got {node.ref_count}"


def test_prefix_cache_ref_count_underflow_prevention():
    """Test that ref_count doesn't go negative."""
    cache = PrefixCacheManager(max_capacity=100)

    # Insert a prefix
    cache.insert((1, 2, 3), slot=0)

    # Release multiple times
    for _ in range(5):
        cache.release((1, 2, 3))

    # Try to find it - should return None since ref_count would be negative
    # or handle it gracefully
    node = cache.root.children.get(1).children.get(2).children.get(3)
    # The ref_count should be 0, not negative
    assert node.ref_count >= 0, f"ref_count went negative: {node.ref_count}"
