import os

import numpy as np
import torch

from astrai.config.train_config import TrainConfig
from astrai.data.serialization import Checkpoint
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.trainer import Trainer


def test_early_stopping_simulation(base_test_env, early_stopping_dataset):
    """Simulate early stopping behavior"""

    def optimizer_fn(model):
        return torch.optim.AdamW(model.parameters())

    def scheduler_fn(optim):
        return SchedulerFactory.create(
            optim, "cosine", warmup_steps=10, lr_decay_steps=10, min_rate=0.05
        )

    train_config = TrainConfig(
        strategy="seq",
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        model=base_test_env["model"],
        dataset=early_stopping_dataset,
        ckpt_dir=base_test_env["test_dir"],
        n_epoch=2,
        batch_size=2,
        ckpt_interval=1,
        accumulation_steps=2,
        random_seed=np.random.randint(1e4),
        device_type=base_test_env["device"],
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
