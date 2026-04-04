import torch

from astrai.config.train_config import TrainConfig
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.train_callback import TrainCallback
from astrai.trainer.trainer import Trainer


def test_callback_integration(base_test_env, random_dataset):
    """Test that all callbacks are properly integrated"""

    def optimizer_fn(model):
        return torch.optim.AdamW(model.parameters())

    def scheduler_fn(optim):
        return SchedulerFactory.create(
            optim, "cosine", warmup_steps=10, lr_decay_steps=10, min_rate=0.05
        )

    train_config = TrainConfig(
        model=base_test_env["model"],
        strategy="seq",
        dataset=random_dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        ckpt_dir=base_test_env["test_dir"],
        n_epoch=1,
        batch_size=2,
        ckpt_interval=3,
        accumulation_steps=1,
        max_grad_norm=1.0,
        random_seed=42,
        device_type=base_test_env["device"],
    )

    # Create custom callbacks to track calls
    callback_calls = []

    class TrackingCallback(TrainCallback):
        def on_train_begin(self, context):
            callback_calls.append("on_train_begin")

        def on_batch_end(self, context):
            callback_calls.append("on_batch_end")

        def on_epoch_end(self, context):
            callback_calls.append("on_epoch_end")

    trainer = Trainer(train_config, callbacks=[TrackingCallback()])

    trainer.train()

    # Verify callbacks were called
    assert "on_train_begin" in callback_calls
    assert "on_batch_end" in callback_calls
    assert "on_epoch_end" in callback_calls
