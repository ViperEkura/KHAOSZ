import torch
import numpy as np

from khaosz.config import *
from khaosz.trainer import *
from khaosz.data.data_util import *

def test_multi_turn_training(base_test_env, multi_turn_dataset):
    """Test training with multi-turn conversation data"""
    optimizer = torch.optim.AdamW(base_test_env["model"].parameters())
    train_config = TrainConfig(
        dataset=multi_turn_dataset,
        optimizer=optimizer,
        checkpoint_dir=base_test_env["test_dir"],
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
        base_test_env["model"], 
        "sft",
        base_test_env["device"],
        bos_token_id=2,
        eos_token_id=3,
        user_token_id=1,
        multi_turn=True
    )
    
    model_parameter = ModelParameter(
        base_test_env["model"], 
        base_test_env["tokenizer"], 
        base_test_env["transformer_config"]
    )
    
    trainer = Trainer(model_parameter, train_config, schedule_config)
    checkpoint = trainer.train()
    
    assert len(checkpoint.loss_list) > 0

def test_schedule_factory_random_configs():
    """Test scheduler factory with random configurations"""
    
    schedule_configs = [
        CosineScheduleConfig(
            warmup_steps=np.random.randint(50, 200),
            total_steps=np.random.randint(1000, 5000),
            min_rate=np.random.uniform(0.01, 0.1)
        ),
        SGDRScheduleConfig(
            warmup_steps=np.random.randint(50, 200),
            cycle_length=np.random.randint(500, 2000),
            t_mult=np.random.randint(1, 3),
            min_rate=np.random.uniform(0.01, 0.1)
        )
    ]
    # todo