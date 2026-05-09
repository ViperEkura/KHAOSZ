"""Tests for scheduler concurrency."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from astrai.inference.scheduler import InferenceScheduler


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
    mock_model.parameters.return_value = iter(
        [MagicMock(dtype=torch.float32, device=torch.device("cpu"))]
    )

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
