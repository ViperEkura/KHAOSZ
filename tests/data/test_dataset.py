import json
import os

import numpy as np
import pytest
import torch

from astrai.dataset.dataset import DatasetFactory, SEQDataset
from astrai.dataset.storage import (
    H5Store,
    MmapStore,
    StoreFactory,
    detect_format,
    json_to_bin,
    load_bin,
    load_json,
    save_bin,
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

    jsonl_path = os.path.join(data_dir, "seq_data.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
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

    jsonl_path = os.path.join(data_dir, "sft_data.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
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

    json_path = os.path.join(data_dir, "data.jsonl")
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


def test_store_unloaded_len():
    """Unloaded Store has __len__ == 0"""
    store = H5Store()
    assert len(store) == 0
    assert store.keys == []


def test_store_fetch_begin_equals_end(base_test_env):
    """Store.fetch with begin == end returns empty tensor"""
    test_dir = base_test_env["test_dir"]
    dummy = {"sequence": [torch.randint(0, 1000, (100,), dtype=torch.int64)]}
    save_h5(test_dir, "empty_fetch", dummy)

    dataset = DatasetFactory.load("seq", test_dir, window_size=32)
    result = dataset.storage.fetch(10, 10, "sequence")
    assert result.numel() == 0


def test_store_empty_data_len(base_test_env):
    """Store loaded with empty data has __len__ == 0"""
    import os

    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "empty_store")
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "data.jsonl"), "w") as f:
        json.dump({"sequence": [[1, 2, 3]]}, f)

    store = StoreFactory.create("json")
    store.load(data_dir)
    assert len(store) > 0

    empty_store = H5Store()
    assert len(empty_store) == 0


def test_store_fetch_before_load():
    """Store.fetch before load raises RuntimeError"""
    store = H5Store()
    with pytest.raises(RuntimeError, match="not loaded"):
        store.fetch(0, 10, "sequence")


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


def test_create_store_invalid_type():
    """StoreFactory.create raises ValueError for unknown type"""
    with pytest.raises(ValueError, match="Unknown component"):
        StoreFactory.create("parquet")


def test_json_pretokenized_without_tokenizer(base_test_env):
    """Pre-tokenized JSON (List[List[int]]) loads without tokenizer"""
    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "json_pretok")
    os.makedirs(data_dir, exist_ok=True)

    json_path = os.path.join(data_dir, "data.jsonl")
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

    with open(os.path.join(test_dir, "data.jsonl"), "w") as f:
        json.dump({"sequence": [[1, 2, 3, 4, 5]]}, f)

    result = load_json(test_dir)
    assert "sequence" in result
    assert "vocab_size" not in result
    assert len(result["sequence"]) == 1


def test_store_multi_segment_concat(base_test_env):
    """Multi-segment H5 data is concatenated into single tensor at load time"""
    import os

    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "multi_seg")
    os.makedirs(data_dir, exist_ok=True)

    segs = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6, 7]),
        torch.tensor([8, 9]),
    ]
    save_h5(data_dir, "data", {"sequence": segs})

    store = StoreFactory.create("h5")
    store.load(data_dir)
    assert len(store) == 9
    result = store.fetch(2, 7, "sequence")
    assert result.tolist() == [3, 4, 5, 6, 7]


def test_save_load_bin_roundtrip(base_test_env):
    """save_bin + load_bin roundtrip preserves data"""
    test_dir = base_test_env["test_dir"]

    data = {
        "sequence": [torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)],
        "loss_mask": [torch.tensor([0, 1, 1, 0, 1], dtype=torch.int64)],
    }
    save_bin(test_dir, data)
    result = load_bin(test_dir)

    assert "sequence" in result
    assert "loss_mask" in result
    assert result["sequence"][0].tolist() == [1, 2, 3, 4, 5]
    assert result["loss_mask"][0].tolist() == [0, 1, 1, 0, 1]


def test_mmap_store_load_and_fetch(base_test_env):
    """MmapStore loads bin data and fetches correctly"""
    test_dir = base_test_env["test_dir"]

    data = {
        "sequence": [torch.randint(0, 1000, (200,), dtype=torch.int64)],
    }
    save_bin(test_dir, data)

    store = StoreFactory.create("bin")
    store.load(test_dir)
    assert len(store) == 200
    assert "sequence" in store.keys

    result = store.fetch(10, 20, "sequence")
    assert result.tolist() == data["sequence"][0][10:20].tolist()


def test_mmap_dataset_load(base_test_env):
    """DatasetFactory.load auto-detects bin format"""
    test_dir = base_test_env["test_dir"]

    data = {
        "sequence": [torch.randint(0, 1000, (200,), dtype=torch.int64)],
    }
    save_bin(test_dir, data)

    dataset = DatasetFactory.load("seq", test_dir, window_size=64)
    assert len(dataset) > 0
    assert dataset.count == 200
    assert dataset[0]["input_ids"].shape[0] == 64


def test_normalize_empty_key():
    """_normalize with empty tensor list does not crash"""
    store = H5Store()
    store._normalize({"sequence": []})
    assert len(store) == 0
    assert store.keys == ["sequence"]


def test_normalize_mixed_empty_key():
    """_normalize with empty + non-empty keys returns min=0"""
    store = H5Store()
    store._normalize({"sequence": [torch.tensor([1, 2, 3])], "loss_mask": []})
    assert len(store) == 0
    assert set(store.keys) == {"sequence", "loss_mask"}


def test_load_jsonl_multiline(base_test_env):
    """JSONL files are loaded line-by-line and accumulated"""
    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "jsonl_test")
    os.makedirs(data_dir, exist_ok=True)

    jsonl_path = os.path.join(data_dir, "data.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write('{"sequence": [[1, 2, 3]]}\n')
        f.write('{"sequence": [[4, 5, 6]]}\n')
        f.write('{"sequence": [[7, 8, 9]]}\n')

    store = StoreFactory.create("json")
    store.load(data_dir)
    assert len(store) == 9
    assert store.fetch(0, 9, "sequence").tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_load_jsonl_with_text_and_tokenizer(base_test_env):
    """JSONL with raw text + tokenizer works"""
    tokenizer = base_test_env["tokenizer"]
    tokenizer_fn = lambda text: tokenizer.encode(text, add_special_tokens=False)

    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "jsonl_text")
    os.makedirs(data_dir, exist_ok=True)

    jsonl_path = os.path.join(data_dir, "data.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write('{"sequence": ["hello world how are you today this is a test"]}\n')

    dataset = DatasetFactory.load(
        "seq", data_dir, window_size=8, tokenizer=tokenizer_fn
    )
    assert len(dataset) > 0
    assert dataset.count > 0


def test_grpo_dataset_dtype(base_test_env):
    """GRPODataset returns correct dtypes"""
    test_dir = base_test_env["test_dir"]

    seq_len = 100
    data = {
        "prompts": [torch.randint(0, 100, (seq_len,), dtype=torch.int32)],
        "responses": [torch.randint(0, 100, (seq_len,), dtype=torch.int32)],
        "masks": [torch.ones(seq_len, dtype=torch.int32)],
        "rewards": [torch.ones(seq_len, dtype=torch.float32)],
    }
    save_h5(test_dir, "grpo_dtype", data)

    dataset = DatasetFactory.load("grpo", test_dir, window_size=32)
    item = dataset[0]

    assert item["prompts"].dtype == torch.long
    assert item["responses"].dtype == torch.long
    assert item["masks"].dtype == torch.bool
    assert item["rewards"].dtype == torch.float32


def test_grpo_dataset_load(base_test_env):
    """GRPODataset loads and returns correct keys"""
    test_dir = base_test_env["test_dir"]
    seq_len = 200
    data = {
        "prompts": [torch.randint(0, 1000, (seq_len,), dtype=torch.int64)],
        "responses": [torch.randint(0, 1000, (seq_len,), dtype=torch.int64)],
        "masks": [torch.ones(seq_len, dtype=torch.int64)],
        "rewards": [torch.rand(seq_len, dtype=torch.float32)],
    }
    save_h5(test_dir, "grpo_test", data)

    dataset = DatasetFactory.load("grpo", test_dir, window_size=64)
    assert len(dataset) > 0
    item = dataset[0]
    assert "prompts" in item
    assert "responses" in item
    assert "masks" in item
    assert "rewards" in item
    assert item["prompts"].shape[0] == 64
    assert item["responses"].shape[0] == 64


def test_detect_format_bin_dir(base_test_env):
    """detect_format returns 'bin' for directory with .bin + meta.json"""
    test_dir = base_test_env["test_dir"]
    save_bin(test_dir, {"sequence": [torch.randint(0, 100, (10,))]})
    assert detect_format(test_dir) == "bin"


def test_detect_format_jsonl_file(base_test_env):
    """detect_format returns 'json' for a single .jsonl file"""
    test_dir = base_test_env["test_dir"]
    path = os.path.join(test_dir, "data.jsonl")
    with open(path, "w") as f:
        f.write('{"sequence": [[1,2,3]]}\n')
    assert detect_format(path) == "json"


def test_store_fetch_multi_key(base_test_env):
    """Store.fetch with List[str] returns Dict[str, Tensor]"""
    test_dir = base_test_env["test_dir"]
    save_h5(
        test_dir,
        "multi_key",
        {
            "sequence": [torch.randint(0, 100, (100,), dtype=torch.int64)],
            "loss_mask": [torch.ones(100, dtype=torch.int64)],
        },
    )

    store = StoreFactory.create("h5")
    store.load(test_dir)
    result = store.fetch(10, 20, ["sequence", "loss_mask"])
    assert isinstance(result, dict)
    assert result["sequence"].shape[0] == 10
    assert result["loss_mask"].shape[0] == 10


def test_store_fetch_out_of_bounds(base_test_env):
    """Store.fetch raises ValueError for out-of-bounds indices"""
    test_dir = base_test_env["test_dir"]
    save_h5(
        test_dir, "bounds", {"sequence": [torch.randint(0, 100, (50,))]}
    )

    store = StoreFactory.create("h5")
    store.load(test_dir)
    with pytest.raises(ValueError, match="out of bounds"):
        store.fetch(-1, 10, "sequence")
    with pytest.raises(ValueError, match="out of bounds"):
        store.fetch(0, 51, "sequence")
    with pytest.raises(ValueError, match="out of bounds"):
        store.fetch(50, 50, "sequence")


def test_json_to_bin_roundtrip(base_test_env):
    """json_to_bin converts JSONL to bin and data is preserved"""
    test_dir = base_test_env["test_dir"]
    jsonl_dir = os.path.join(test_dir, "src")
    os.makedirs(jsonl_dir, exist_ok=True)

    with open(os.path.join(jsonl_dir, "data.jsonl"), "w") as f:
        f.write('{"sequence": [[1, 2, 3, 4, 5]]}\n')

    bin_dir = os.path.join(test_dir, "out")
    json_to_bin(jsonl_dir, bin_dir)

    store = StoreFactory.create("bin")
    store.load(bin_dir)
    assert len(store) == 5
    assert store.fetch(0, 5, "sequence").tolist() == [1, 2, 3, 4, 5]


def test_dpo_dataset_from_jsonl(base_test_env):
    """DPO dataset loaded from pre-tokenized JSONL"""
    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "dpo_jsonl")
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "dpo.jsonl"), "w") as f:
        f.write(
            json.dumps(
                {
                    "chosen": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10],
                    "rejected": [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 10],
                    "chosen_mask": [[1] * 100],
                    "rejected_mask": [[1] * 100],
                }
            )
            + "\n"
        )

    dataset = DatasetFactory.load("dpo", data_dir, window_size=32)
    assert len(dataset) > 0
    item = dataset[0]
    assert item["chosen"].dtype == torch.long
    assert item["rejected"].dtype == torch.long
    assert item["chosen_mask"].dtype == torch.bool
    assert item["rejected_mask"].dtype == torch.bool


def test_dataset_load_explicit_storage_type(base_test_env):
    """DatasetFactory.load with explicit storage_type bypasses auto-detect"""
    test_dir = base_test_env["test_dir"]
    save_h5(
        test_dir, "explicit", {"sequence": [torch.randint(0, 100, (200,))]}
    )

    dataset = DatasetFactory.load(
        "seq", test_dir, window_size=64, storage_type="h5"
    )
    assert len(dataset) > 0
    assert dataset.count == 200
