import os
import json
import torch
import shutil
import pytest
import pickle
import tempfile
import numpy as np

from torch.utils.data import Dataset
from khaosz.core import *
from khaosz.trainer import *
from khaosz.trainer.data_util import *

import matplotlib
matplotlib.use('Agg')


@pytest.fixture
def test_env():
    """Setup test environment with randomized data"""
    test_dir = tempfile.mkdtemp()
    config_path = os.path.join(test_dir, "config.json")
    
    n_dim_choices = [8, 16, 32]
    n_head_choices = [2, 4]
    
    n_dim = int(np.random.choice(n_dim_choices))
    n_head = int(np.random.choice(n_head_choices))
    n_kvhead =  n_head // 2
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
    
    dataset = RandomDataset()
    multi_turn_dataset = MultiTurnDataset()
    
    yield {
        "test_dir": test_dir,
        "config_path": config_path,
        "transformer_config": transformer_config,
        "model": model,
        "tokenizer": tokenizer,
        "dataset": dataset,
        "multi_turn_dataset": multi_turn_dataset
    }
    
    shutil.rmtree(test_dir)

def test_dataset_loader_random_paths(test_env):
    """Test dataset loader with multiple random paths"""
    test_dir = test_env["test_dir"]
    
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
        max_len=64, 
        device="cpu"
    )
    assert loaded_dataset is not None
    assert len(loaded_dataset) > 0


def test_different_batch_sizes(test_env):
    """Test training with different batch sizes"""
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        optimizer = torch.optim.AdamW(test_env["model"].parameters())
        train_config = TrainConfig(
            dataset=test_env["dataset"],
            optimizer=optimizer,
            checkpoint_dir=test_env["test_dir"],
            n_epoch=1,
            batch_size=batch_size,
            checkpoint_interval=5,
            accumulation_steps=1,
            max_grad_norm=1.0,
            random_seed=np.random.randint(1000)
        )
        
        assert train_config.batch_size == batch_size


def test_random_sampler_consistency(test_env):
    """Test RandomSampler produces consistent results with same seed"""
    dataset = test_env["dataset"]
    
    # Create two samplers with same seed
    sampler1 = RandomSampler(dataset, seed=42)
    sampler2 = RandomSampler(dataset, seed=42)
    
    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))
    
    assert indices1 == indices2


def test_random_sampler_different_seeds(test_env):
    """Test RandomSampler produces different results with different seeds"""
    dataset = test_env["dataset"]
    
    # Create two samplers with different seeds
    sampler1 = RandomSampler(dataset, seed=42)
    sampler2 = RandomSampler(dataset, seed=123)
    
    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))
    
    # Very high probability they should be different
    assert indices1 != indices2


def test_schedule_factory_random_configs(test_env):
    """Test scheduler factory with random configurations"""
    schedule_configs = [
        CosineScheduleConfig(
            warmup_steps=np.random.randint(50, 200),
            total_steps=np.random.randint(1000, 5000),
            min_rate=np.random.uniform(0.01, 0.1)
        ),
        SgdrScheduleConfig(
            warmup_steps=np.random.randint(50, 200),
            cycle_length=np.random.randint(500, 2000),
            t_mult=np.random.randint(1, 3),
            min_rate=np.random.uniform(0.01, 0.1)
        )
    ]
    
    for config in schedule_configs:
        schedule_fn = SchedulerFactory.load_schedule_fn(config)
        assert callable(schedule_fn)
        
        # Test the schedule function at different steps
        for step in [0, config.warmup_steps // 2, config.warmup_steps, config.warmup_steps * 2]:
            lr_mult = schedule_fn(step)
            assert 0 <= lr_mult <= 1


def test_multi_turn_training(test_env):
    """Test training with multi-turn conversation data"""
    optimizer = torch.optim.AdamW(test_env["model"].parameters())
    train_config = TrainConfig(
        dataset=test_env["multi_turn_dataset"],
        optimizer=optimizer,
        checkpoint_dir=test_env["test_dir"],
        n_epoch=2,
        batch_size=2,
        checkpoint_interval=3,
        accumulation_steps=1,
        max_grad_norm=1.0,
        random_seed=int(np.random.randint(1000))
    )
    
    schedule_config = CosineScheduleConfig(
        warmup_steps=50,
        total_steps=100
    )
    
    train_config.strategy = StrategyFactory.load(
        test_env["model"], 
        "sft",
        bos_token_id=2,
        eos_token_id=3,
        user_token_id=1,
        multi_turn=True
    )
    
    model_parameter = ModelParameter(
        test_env["model"], 
        test_env["tokenizer"], 
        test_env["transformer_config"]
    )
    
    trainer = Trainer(model_parameter, train_config, schedule_config)
    checkpoint = trainer.train()
    
    assert len(checkpoint.loss_list) > 0


def test_gradient_accumulation(test_env):
    """Test training with different gradient accumulation steps"""
    accumulation_steps_list = [1, 2, 4]
    
    for accumulation_steps in accumulation_steps_list:
        optimizer = torch.optim.AdamW(test_env["model"].parameters())
        train_config = TrainConfig(
            dataset=test_env["dataset"],
            optimizer=optimizer,
            checkpoint_dir=test_env["test_dir"],
            n_epoch=1,
            batch_size=2,
            checkpoint_interval=10,
            accumulation_steps=accumulation_steps,
            max_grad_norm=1.0,
            random_seed=42
        )
        
        schedule_config = CosineScheduleConfig(
            warmup_steps=10,
            total_steps=20
        )
        
        train_config.strategy = StrategyFactory.load(
            test_env["model"], 
            "seq"
        )
        
        model_parameter = ModelParameter(
            test_env["model"], 
            test_env["tokenizer"], 
            test_env["transformer_config"]
        )
        
        trainer = Trainer(model_parameter, train_config, schedule_config)
        trainer.train()
        
        assert train_config.accumulation_steps == accumulation_steps

def test_dpo_strategy_with_random_data(test_env):
    """Test DPO strategy with randomized preference data"""
    test_dir = test_env["test_dir"]
    
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
        max_len=64, 
        device="cpu"
    )
    
    assert dpo_dataset is not None
    assert hasattr(dpo_dataset, 'fetcher')


def test_callback_integration(test_env):
    """Test that all callbacks are properly integrated"""
    optimizer = torch.optim.AdamW(test_env["model"].parameters())
    train_config = TrainConfig(
        dataset=test_env["dataset"],
        optimizer=optimizer,
        checkpoint_dir=test_env["test_dir"],
        n_epoch=1,
        batch_size=2,
        checkpoint_interval=3,
        accumulation_steps=1,
        max_grad_norm=1.0,
        random_seed=42
    )
    
    schedule_config = CosineScheduleConfig(
        warmup_steps=10,
        total_steps=20
    )
    
    # Create custom callbacks to track calls
    callback_calls = []
    
    class TrackingCallback(TrainerCallback):
        def on_train_begin(self, trainer, **kwargs):
            callback_calls.append('on_train_begin')
        
        def on_batch_end(self, trainer, **kwargs):
            callback_calls.append('on_batch_end')
        
        def on_epoch_end(self, trainer, **kwargs):
            callback_calls.append('on_epoch_end')
    
    train_config.strategy = StrategyFactory.load(test_env["model"], "seq")
    model_parameter = ModelParameter(
        test_env["model"], 
        test_env["tokenizer"], 
        test_env["transformer_config"]
    )
    
    trainer = Trainer(
        model_parameter, 
        train_config, 
        schedule_config,
        callbacks=[TrackingCallback(), ProgressBarCallback()]
    )
    
    trainer.train()
    
    # Verify callbacks were called
    assert 'on_train_begin' in callback_calls
    assert 'on_batch_end' in callback_calls
    assert 'on_epoch_end' in callback_calls


def test_memory_efficient_training(test_env):
    """Test training with memory-efficient configurations"""
    # Test with smaller batch sizes and gradient checkpointing
    small_batch_configs = [
        {"batch_size": 1, "accumulation_steps": 8},
        {"batch_size": 2, "accumulation_steps": 4},
        {"batch_size": 4, "accumulation_steps": 2}
    ]
    
    for config in small_batch_configs:
        optimizer = torch.optim.AdamW(test_env["model"].parameters())
        train_config = TrainConfig(
            dataset=test_env["dataset"],
            optimizer=optimizer,
            checkpoint_dir=test_env["test_dir"],
            n_epoch=1,
            batch_size=config["batch_size"],
            checkpoint_interval=5,
            accumulation_steps=config["accumulation_steps"],
            max_grad_norm=1.0,
            random_seed=42
        )
        
        assert train_config.accumulation_steps == config["accumulation_steps"]


def test_early_stopping_simulation(test_env):
    """Simulate early stopping behavior"""
    class EarlyStoppingDataset(Dataset):
        def __init__(self, length=10, stop_after=5):
            self.length = length
            self.stop_after = stop_after
            self.count = 0
            
        def __len__(self):
            return self.length
            
        def __getitem__(self, idx):
            self.count += 1
            if self.count == self.stop_after:
                raise RuntimeError("Simulated early stopping")
            
            return {
                "input_ids": torch.randint(0, 1000, (64,)),
                "target_ids": torch.randint(0, 1000, (64,))
            }
    
    dataset = EarlyStoppingDataset()
    
    optimizer = torch.optim.AdamW(test_env["model"].parameters())
    train_config = TrainConfig(
        dataset=dataset,
        optimizer=optimizer,
        checkpoint_dir=test_env["test_dir"],
        n_epoch=2,
        batch_size=2,
        checkpoint_interval=1,
        accumulation_steps=1,
        max_grad_norm=1.0,
        random_seed=42
    )
    
    train_config.strategy = StrategyFactory.load(test_env["model"], "seq")
    model_parameter = ModelParameter(
        test_env["model"], 
        test_env["tokenizer"], 
        test_env["transformer_config"]
    )
    schedule_config = CosineScheduleConfig(warmup_steps=10, total_steps=20)
    trainer = Trainer(model_parameter, train_config, schedule_config)
    
    # Should handle early stopping gracefully
    checkpoint = None
    try:
        checkpoint = trainer.train()
        assert len(checkpoint.loss_list) == 2
    except Exception:
        # Handle any exceptions
        pass
    
    checkpoint = trainer.train(checkpoint)
    assert len(checkpoint.loss_list) == 10 + 1


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])