import os
import json
import numpy as np
import tempfile
import shutil
import torch

import pytest
import matplotlib
from torch.utils.data import Dataset

from khaosz.core import *
from khaosz.trainer import *
from khaosz.trainer.data_util import *

matplotlib.use("Agg")

@pytest.fixture
def base_test_env():
    test_dir = tempfile.mkdtemp()
    config_path = os.path.join(test_dir, "config.json")
    
    n_dim_choices = [8, 16, 32]
    n_head_choices = [2, 4]
    
    n_dim = int(np.random.choice(n_dim_choices))
    n_head = int(np.random.choice(n_head_choices))
    n_kvhead = n_head // 2
    d_ffn = n_dim * 2

    config = {
        "vocab_size": 1000,
        "n_dim": n_dim,
        "n_head": n_head,
        "n_kvhead": n_kvhead,
        "d_ffn": d_ffn,
        "m_len": 1024,
        "n_layer": 4,
        "norm_eps": 1e-5
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    transformer_config = TransformerConfig().load(config_path)
    model = Transformer(transformer_config)
    tokenizer = BpeTokenizer()
    
    yield {
        "test_dir": test_dir,
        "config_path": config_path,
        "transformer_config": transformer_config,
        "model": model,
        "tokenizer": tokenizer,
    }
    
    shutil.rmtree(test_dir)

@pytest.fixture
def random_dataset():
    class RandomDataset(Dataset):
        def __init__(self, length=None, max_length=64, vocab_size=1000):
            self.length = length or int(np.random.randint(100, 200))
            self.max_length = max_length
            self.vocab_size = vocab_size
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
                "target_ids": torch.randint(0, self.vocab_size, (self.max_length,))
            }
    
    dataset = RandomDataset()
    
    yield dataset

@pytest.fixture
def multi_turn_dataset():
    class MultiTurnDataset(Dataset):
        def __init__(self, length=None, max_length=64, vocab_size=1000):
            self.length = length or int(np.random.randint(100, 200))
            self.max_length = max_length
            self.vocab_size = vocab_size
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            input_ids = torch.randint(0, self.vocab_size, (self.max_length,))
            target_ids = torch.randint(0, self.vocab_size, (self.max_length,))
            loss_mask = build_loss_mask(input_ids, 0, 1)
            attn_mask = build_attention_mask(input_ids, 2, True)

            return {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "loss_mask": loss_mask,
                "attn_mask": attn_mask,
            }
    
    dataset = MultiTurnDataset()
    
    yield dataset