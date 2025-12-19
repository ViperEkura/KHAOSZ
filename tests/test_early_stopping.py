import os
import torch
import numpy as np
from khaosz.config import *
from khaosz.trainer import *


def test_early_stopping_simulation(base_test_env, early_stopping_dataset):
    """Simulate early stopping behavior"""
    
    schedule_config = CosineScheduleConfig(warmup_steps=10, total_steps=20)
    optimizer = torch.optim.AdamW(base_test_env["model"].parameters())
    scheduler = SchedulerFactory.load(optimizer, schedule_config)
    
    train_config = TrainConfig(
        strategy="seq",
        scheduler=scheduler,
        model=base_test_env["model"],
        dataset=early_stopping_dataset,
        optimizer=optimizer,
        checkpoint_dir=base_test_env["test_dir"],
        n_epoch=2,
        batch_size=2,
        checkpoint_interval=1,
        accumulation_steps=2,
        random_seed=np.random.randint(1e4),
    )

    trainer = Trainer(train_config)
    
    # Should handle early stopping gracefully
    checkpoint = None
    try:
        checkpoint = trainer.train()
    except Exception:
        # Handle any exceptions
        pass
    
    load_dir = os.path.join(base_test_env["test_dir"], "epoch_0_iter_2")
    checkpoint = Checkpoint.load(load_dir)
    trainer.train(checkpoint)
    
    load_dir = os.path.join(base_test_env["test_dir"], "epoch_1_iter_10")
    checkpoint = Checkpoint.load(load_dir)
    assert checkpoint.iteration == 10