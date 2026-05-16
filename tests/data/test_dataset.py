import json
import os

import numpy as np
import pytest
import torch

from astrai.dataset.dataset import DatasetFactory, SEQDataset
from astrai.dataset.storage import (
    BaseSegmentFetcher,
    H5Storage,
    MultiSegmentFetcher,
    StorageFactory,
    detect_format,
    load_json,
    save_h5,
)


def test_dataset_loader_random_paths(base_test_env):
    """Test dataset loader with multiple random paths"""
    test_dir = base_test_env["test_dir"]

    # Create multiple mmap dataset directories with random data
    num_files = np.random.randint(2, 5)

    for i in range(num_files):
        seq_length = np.random.randint(200, 400)
        dummy_data = {
            "sequence": [
                torch.randint(0, 1000, (seq_length,), dtype=torch.int64)
                for _ in range(10)
            ],
        }
        save_h5(test_dir, f"data_{i}", dummy_data)

        # Test loading with multiple paths
        loaded_dataset = DatasetFactory.load(
            train_type="seq",
            load_path=test_dir,
            window_size=64,
        )
        assert loaded_dataset is not None
        assert len(loaded_dataset) > 0

    # Test that we can get items without errors
    for i in range(len(loaded_dataset)):
        item = loaded_dataset[i]
        assert "input_ids" in item
        assert "target_ids" in item
        assert item["input_ids"].shape == item["target_ids"].shape
        assert item["input_ids"].shape[0] == 64


def test_dpo_strategy_with_random_data(base_test_env):
    """Test DPO strategy with randomized preference data"""
    test_dir = base_test_env["test_dir"]

    # Create DPO-style data with memory mapping format
    seq_length = np.random.randint(100, 200)

    dummy_data = {
        "chosen": [torch.randint(0, 1000, (seq_length,), dtype=torch.int64)],
        "rejected": [torch.randint(0, 1000, (seq_length,), dtype=torch.int64)],
        "chosen_mask": [torch.ones(seq_length, dtype=torch.bool)],
        "rejected_mask": [torch.ones(seq_length, dtype=torch.bool)],
    }

    save_h5(test_dir, "dpo_data", dummy_data)

    # Load DPO dataset
    dpo_dataset = DatasetFactory.load(
        train_type="dpo",
        load_path=test_dir,
        window_size=64,
    )

    assert dpo_dataset is not None
    assert dpo_dataset.storage is not None
    assert len(dpo_dataset) > 0

    # Test that we can get DPO items without errors
    for i in range(min(3, len(dpo_dataset))):
        item = dpo_dataset[i]
        assert "chosen" in item
        assert "rejected" in item
        assert "chosen_mask" in item
        assert "rejected_mask" in item
        assert item["chosen"].shape == item["rejected"].shape
        assert item["chosen_mask"].shape == item["rejected_mask"].shape


def test_sft_dataset_with_random_data(base_test_env):
    """Test SFT dataset with random data"""
    test_dir = base_test_env["test_dir"]

    # Create SFT-style data with memory mapping format
    seq_length = np.random.randint(100, 200)

    dummy_data = {
        "sequence": [torch.randint(0, 1000, (seq_length,), dtype=torch.int64)],
        "loss_mask": [torch.ones(seq_length, dtype=torch.bool)],
    }

    save_h5(test_dir, "sft_data", dummy_data)

    # Load SFT dataset
    sft_dataset = DatasetFactory.load(
        train_type="sft",
        load_path=test_dir,
        window_size=64,
    )

    assert sft_dataset is not None
    assert sft_dataset.storage is not None
    assert len(sft_dataset) > 0

    # Test that we can get SFT items without errors
    for i in range(min(3, len(sft_dataset))):
        item = sft_dataset[i]
        assert "input_ids" in item
        assert "target_ids" in item
        assert "loss_mask" in item
        assert item["input_ids"].shape == item["target_ids"].shape
        assert item["loss_mask"].shape[0] == 64


def test_dataset_with_custom_stride(base_test_env):
    """Test dataset with custom stride parameter"""
    test_dir = base_test_env["test_dir"]

    # Create test data
    seq_length = 200
    dummy_data = {
        "sequence": [torch.randint(0, 1000, (seq_length,), dtype=torch.int64)],
    }

    save_h5(test_dir, "stride_test_data", dummy_data)

    # Test with custom stride
    custom_stride = 32
    dataset = DatasetFactory.load(
        train_type="seq", load_path=test_dir, window_size=64, stride=custom_stride
    )

    assert dataset is not None
    assert len(dataset) > 0

    # With stride 32 and window 64 on 200 length data, we should get more samples
    # than with default stride (which equals window size)
    default_stride_dataset = DatasetFactory.load(
        train_type="seq",
        load_path=test_dir,
        window_size=64,
    )

    assert len(dataset) > len(default_stride_dataset)


# ============== JSON Storage Tests (raw text + tokenizer) ==============


def _make_tokenizer_fn(tokenizer):
    """Wrap tokenizer.encode() as a str -> List[int] callable."""
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


def test_seq_dataset_from_json_text(base_test_env):
    """Test loading SEQ dataset from raw-text JSON with tokenizer"""
    tokenizer = base_test_env["tokenizer"]
    tokenizer_fn = _make_tokenizer_fn(tokenizer)
    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "json_text")
    os.makedirs(data_dir, exist_ok=True)

    texts = [
        "hello world this is a test sentence for tokenizer",
        "another sentence with different words and tokens",
        "machine learning is fascinating and powerful",
    ]

    json_path = os.path.join(data_dir, "seq_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"sequence": texts}, f, ensure_ascii=False)

    dataset = DatasetFactory.load(
        train_type="seq",
        load_path=data_dir,
        window_size=16,
        tokenizer=tokenizer_fn,
    )
    assert dataset is not None
    assert len(dataset) > 0
    assert dataset.count > 0
    assert "sequence" in dataset.keys

    item = dataset[0]
    assert "input_ids" in item
    assert "target_ids" in item
    assert item["input_ids"].shape[0] == 16


def test_sft_dataset_from_json_text(base_test_env):
    """Test loading SFT dataset from raw-text JSON with tokenizer"""
    tokenizer = base_test_env["tokenizer"]
    tokenizer_fn = _make_tokenizer_fn(tokenizer)
    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "json_sft")
    os.makedirs(data_dir, exist_ok=True)

    texts = [
        "user asks a question about the weather",
        "assistant provides a helpful response to the user",
    ]

    json_path = os.path.join(data_dir, "sft_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"sequence": texts, "loss_mask": texts},
            f,
            ensure_ascii=False,
        )

    dataset = DatasetFactory.load(
        train_type="sft",
        load_path=data_dir,
        window_size=16,
        tokenizer=tokenizer_fn,
    )
    assert dataset is not None
    assert len(dataset) > 0

    item = dataset[0]
    assert "loss_mask" in item


def test_json_storage_explicit_tokenizer(base_test_env):
    """Test explicit JSON storage with tokenizer"""
    tokenizer = base_test_env["tokenizer"]
    tokenizer_fn = _make_tokenizer_fn(tokenizer)
    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "json_explicit")
    os.makedirs(data_dir, exist_ok=True)

    texts = ["abcdefghijklmnopqrstuvwxyz" * 10]

    json_path = os.path.join(data_dir, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"sequence": texts}, f, ensure_ascii=False)

    token_count = len(tokenizer_fn(texts[0]))

    dataset = DatasetFactory.load(
        train_type="seq",
        load_path=data_dir,
        window_size=32,
        storage_type="json",
        tokenizer=tokenizer_fn,
    )
    assert dataset is not None
    assert len(dataset) > 0
    assert dataset.count == token_count


def test_dataset_count_property(base_test_env):
    """Test the count property returns correct raw token count"""
    test_dir = base_test_env["test_dir"]

    seq_length = 200
    dummy_data = {
        "sequence": [torch.randint(0, 1000, (seq_length,), dtype=torch.int64)],
    }

    save_h5(test_dir, "count_test_data", dummy_data)

    dataset = DatasetFactory.load(
        train_type="seq",
        load_path=test_dir,
        window_size=64,
    )

    assert dataset.count == seq_length
    assert dataset.count > len(dataset)  # raw tokens > windows
    assert len(dataset) == (seq_length - 1 - 64) // 64 + 1


def test_empty_dataset_count():
    """Test count returns 0 when no data is loaded"""
    dataset = SEQDataset(window_size=64, stride=32)
    assert dataset.count == 0
    assert dataset.keys == []


def test_dataset_too_short_for_window(base_test_env):
    """Dataset shorter than window_size returns __len__ == 0"""
    test_dir = base_test_env["test_dir"]
    seq_length = 30
    save_h5(
        test_dir,
        "short",
        {"sequence": [torch.randint(0, 1000, (seq_length,), dtype=torch.int64)]},
    )
    dataset = DatasetFactory.load("seq", test_dir, window_size=64)
    assert len(dataset) == 0
    assert dataset.count == seq_length


def test_unloaded_dataset_getitem_raises():
    """__getitem__ without load() should fail clearly"""
    dataset = SEQDataset(window_size=64, stride=32)
    with pytest.raises(RuntimeError, match="not loaded"):
        dataset.get_index(0)


def test_unloaded_dataset_len():
    """__len__ without load() returns 0"""
    dataset = SEQDataset(window_size=64, stride=32)
    assert len(dataset) == 0


def test_base_segment_fetcher_empty():
    """BaseSegmentFetcher with empty segments list"""
    fetcher = BaseSegmentFetcher([])
    assert len(fetcher) == 0
    with pytest.raises(ValueError, match="out of bounds"):
        fetcher.fetch_data(0, 1)


def test_base_segment_fetcher_begin_equals_end(base_test_env):
    """fetch_data with begin == end returns empty tensor"""
    test_dir = base_test_env["test_dir"]
    dummy = {"sequence": [torch.randint(0, 1000, (100,), dtype=torch.int64)]}
    save_h5(test_dir, "empty_fetch", dummy)

    dataset = DatasetFactory.load("seq", test_dir, window_size=32)
    fetcher = dataset.storage._fetcher.multi_fetchers["sequence"]
    result = fetcher.fetch_data(10, 10)
    assert result.numel() == 0


def test_multi_segment_fetcher_empty_dict():
    """MultiSegmentFetcher with empty dict has __len__ == 0"""
    fetcher = MultiSegmentFetcher({})
    assert len(fetcher) == 0


def test_storage_fetch_before_load():
    """BaseStorage.fetch before load raises RuntimeError"""
    storage = H5Storage()
    with pytest.raises(RuntimeError, match="not loaded"):
        storage.fetch(0, 10, "sequence")


def test_detect_format_nonexistent_path():
    """detect_format raises FileNotFoundError for bad path"""
    with pytest.raises(FileNotFoundError, match="No supported"):
        detect_format("/nonexistent/path/xyz")


def test_detect_format_unsupported_file(base_test_env):
    """detect_format raises ValueError for unsupported file extension"""
    test_dir = base_test_env["test_dir"]
    path = os.path.join(test_dir, "data.txt")
    with open(path, "w") as f:
        f.write("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        detect_format(path)


def test_create_storage_invalid_type():
    """StorageFactory.create raises ValueError for unknown type"""
    with pytest.raises(ValueError, match="Unknown component"):
        StorageFactory.create("parquet")


def test_json_pretokenized_without_tokenizer(base_test_env):
    """Pre-tokenized JSON (List[List[int]]) loads without tokenizer"""
    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "json_pretok")
    os.makedirs(data_dir, exist_ok=True)

    json_path = os.path.join(data_dir, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"sequence": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]}, f)

    dataset = DatasetFactory.load("seq", data_dir, window_size=4, storage_type="json")
    assert len(dataset) > 0
    assert dataset.count == 10

    item = dataset[0]
    assert item["input_ids"].tolist() == [1, 2, 3, 4]
    assert item["target_ids"].tolist() == [2, 3, 4, 5]


def test_load_json_skips_config_file(base_test_env):
    """load_json skips scalar-value config files"""
    test_dir = base_test_env["test_dir"]
    with open(os.path.join(test_dir, "config.json"), "w") as f:
        json.dump({"vocab_size": 1000, "dim": 16}, f)

    with open(os.path.join(test_dir, "data.json"), "w") as f:
        json.dump({"sequence": [[1, 2, 3, 4, 5]]}, f)

    result = load_json(test_dir)
    assert "sequence" in result
    assert "vocab_size" not in result
    assert len(result["sequence"]) == 1


def test_base_segment_fetcher_multi_segment():
    """fetch_data across multiple segment boundaries"""
    segs = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6, 7]),
        torch.tensor([8, 9]),
    ]
    fetcher = BaseSegmentFetcher(segs)
    assert len(fetcher) == 9
    result = fetcher.fetch_data(2, 7)
    assert result.tolist() == [3, 4, 5, 6, 7]
