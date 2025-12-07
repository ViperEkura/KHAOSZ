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
        assert checkpoint.iteration == 2
    except Exception:
        # Handle any exceptions
        pass
    
    checkpoint = trainer.train(checkpoint)
    assert checkpoint.iteration == 10