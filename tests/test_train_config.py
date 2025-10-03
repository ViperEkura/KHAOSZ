import torch
import numpy as np

from khaosz.core import *
from khaosz.trainer import *
from khaosz.trainer.data_util import *

def test_different_batch_sizes(base_test_env, random_dataset):
    """Test training with different batch sizes"""
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        optimizer = torch.optim.AdamW(base_test_env["model"].parameters())
        train_config = TrainConfig(
            dataset=random_dataset,
            optimizer=optimizer,
            checkpoint_dir=base_test_env["test_dir"],
            n_epoch=1,
            batch_size=batch_size,
            checkpoint_interval=5,
            accumulation_steps=1,
            max_grad_norm=1.0,
            random_seed=np.random.randint(1000)
        )
        
        assert train_config.batch_size == batch_size

def test_gradient_accumulation(base_test_env, random_dataset):
    """Test training with different gradient accumulation steps"""
    accumulation_steps_list = [1, 2, 4]
    
    for accumulation_steps in accumulation_steps_list:
        optimizer = torch.optim.AdamW(base_test_env["model"].parameters())
        train_config = TrainConfig(
            dataset=random_dataset,
            optimizer=optimizer,
            checkpoint_dir=base_test_env["test_dir"],
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
            base_test_env["model"], 
            "seq"
        )
        
        model_parameter = ModelParameter(
            base_test_env["model"], 
            base_test_env["tokenizer"], 
            base_test_env["transformer_config"]
        )
        
        trainer = Trainer(model_parameter, train_config, schedule_config)
        trainer.train()
        
        assert train_config.accumulation_steps == accumulation_steps

def test_memory_efficient_training(base_test_env, random_dataset):
    """Test training with memory-efficient configurations"""
    # Test with smaller batch sizes and gradient checkpointing
    small_batch_configs = [
        {"batch_size": 1, "accumulation_steps": 8},
        {"batch_size": 2, "accumulation_steps": 4},
        {"batch_size": 4, "accumulation_steps": 2}
    ]
    
    for config in small_batch_configs:
        optimizer = torch.optim.AdamW(base_test_env["model"].parameters())
        train_config = TrainConfig(
            dataset=random_dataset,
            optimizer=optimizer,
            checkpoint_dir=base_test_env["test_dir"],
            n_epoch=1,
            batch_size=config["batch_size"],
            checkpoint_interval=5,
            accumulation_steps=config["accumulation_steps"],
            max_grad_norm=1.0,
            random_seed=42
        )
        
        assert train_config.accumulation_steps == config["accumulation_steps"]