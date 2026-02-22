import os
import torch
import numpy as np

from khaosz.data.file import save_h5
from khaosz.data.dataset import *


def create_h5_dataset(dir_path, data_dict, dataset_name):
    """Helper function to create HDF5 dataset for testing"""
    dataset_path = os.path.join(dir_path, f"{dataset_name}.h5")
    
    # Convert data_dict to the format expected by save_h5
    # save_h5 expects a list of tensors for each key
    tensor_group = {key: [tensor] for key, tensor in data_dict.items()}
    
    save_h5(dataset_path, tensor_group)
    
    return dataset_path


def test_dataset_loader_random_paths(base_test_env):
    """Test dataset loader with multiple random paths"""
    test_dir = base_test_env["test_dir"]
    
    # Create multiple mmap dataset directories with random data
    num_files = np.random.randint(2, 5)
    
    for i in range(num_files):
        seq_length = np.random.randint(100, 200)
        dummy_data = {
            "sequence": torch.randint(0, 1000, (seq_length,), dtype=torch.int64),
        }
        dataset_path = create_h5_dataset(test_dir, dummy_data, f"test_data_{i}")
    
        # Test loading with multiple paths
        loaded_dataset = DatasetLoader.load(
            train_type="seq", 
            load_path=dataset_path, 
            window_size=64, 
        )
        assert loaded_dataset is not None
        assert len(loaded_dataset) > 0
    
    # Test that we can get items without errors
    for i in range(min(3, len(loaded_dataset))):
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
        "chosen": torch.randint(0, 1000, (seq_length,), dtype=torch.int64),
        "rejected": torch.randint(0, 1000, (seq_length,), dtype=torch.int64),
        "chosen_mask": torch.ones(seq_length, dtype=torch.bool),
        "rejected_mask": torch.ones(seq_length, dtype=torch.bool)
    }
    
    dataset_path = create_h5_dataset(test_dir, dummy_data, "dpo_data")
    
    # Load DPO dataset
    dpo_dataset = DatasetLoader.load(
        train_type="dpo", 
        load_path=dataset_path, 
        window_size=64, 
    )
    
    assert dpo_dataset is not None
    assert hasattr(dpo_dataset, 'fetcher')
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
        "sequence": torch.randint(0, 1000, (seq_length,), dtype=torch.int64),
        "loss_mask": torch.ones(seq_length, dtype=torch.bool)
    }
    
    dataset_path = create_h5_dataset(test_dir, dummy_data, "sft_data")
    
    # Load SFT dataset
    sft_dataset = DatasetLoader.load(
        train_type="sft", 
        load_path=dataset_path, 
        window_size=64, 
    )
    
    assert sft_dataset is not None
    assert hasattr(sft_dataset, 'fetcher')
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
        "sequence": torch.randint(0, 1000, (seq_length,), dtype=torch.int64),
    }
    
    dataset_path = create_h5_dataset(test_dir, dummy_data, "stride_test_data")
    
    # Test with custom stride
    custom_stride = 32
    dataset = DatasetLoader.load(
        train_type="seq", 
        load_path=dataset_path, 
        window_size=64, 
        stride=custom_stride
    )
    
    assert dataset is not None
    assert len(dataset) > 0
    
    # With stride 32 and window 64 on 200 length data, we should get more samples
    # than with default stride (which equals window size)
    default_stride_dataset = DatasetLoader.load(
        train_type="seq", 
        load_path=dataset_path, 
        window_size=64, 
    )
    
    assert len(dataset) > len(default_stride_dataset)
