import os
import torch
import pickle
import numpy as np

from khaosz.trainer import *
from khaosz.data.data_util import *


def test_dataset_loader_random_paths(base_test_env):
    """Test dataset loader with multiple random paths"""
    test_dir = base_test_env["test_dir"]
    
    # Create multiple pkl files with random data
    num_files = np.random.randint(2, 5)
    pkl_paths = []
    
    for i in range(num_files):
        pkl_path = os.path.join(test_dir, f"test_data_{i}.pkl")
        seq_length = np.random.randint(50, 100)
        dummy_data = {
            "sequence": torch.randint(0, 1000, (seq_length,)),
            "chosen": torch.randint(0, 1000, (seq_length,)),
            "rejected": torch.randint(0, 1000, (seq_length,)),
            "chosen_mask": torch.ones(seq_length, dtype=torch.bool),
            "rejected_mask": torch.ones(seq_length, dtype=torch.bool)
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(dummy_data, f)
        pkl_paths.append(pkl_path)
    
    # Test loading with multiple paths
    loaded_dataset = DatasetLoader.load(
        train_type="seq", 
        load_path=pkl_paths, 
        window_size=64, 
    )
    assert loaded_dataset is not None
    assert len(loaded_dataset) > 0

def test_dpo_strategy_with_random_data(base_test_env):
    """Test DPO strategy with randomized preference data"""
    test_dir = base_test_env["test_dir"]
    
    # Create DPO-style data
    pkl_path = os.path.join(test_dir, "dpo_data.pkl")
    seq_length = np.random.randint(40, 80)
    
    dummy_data = {
        "chosen": torch.randint(0, 1000, (seq_length,)),
        "rejected": torch.randint(0, 1000, (seq_length,)),
        "chosen_mask": torch.ones(seq_length, dtype=torch.bool),
        "rejected_mask": torch.ones(seq_length, dtype=torch.bool)
    }
    
    with open(pkl_path, "wb") as f:
        pickle.dump(dummy_data, f)
    
    # Load DPO dataset
    dpo_dataset = DatasetLoader.load(
        train_type="dpo", 
        load_path=pkl_path, 
        window_size=64, 
    )
    
    assert dpo_dataset is not None
    assert hasattr(dpo_dataset, 'fetcher')