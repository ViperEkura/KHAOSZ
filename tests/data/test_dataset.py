import numpy as np
import torch

from astrai.data.dataset import DatasetFactory
from astrai.data.serialization import save_h5


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
    assert hasattr(dpo_dataset, "fetcher")
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
    assert hasattr(sft_dataset, "fetcher")
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
