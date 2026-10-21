import torch

from astrai.config import ComponentSpec, TrainConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.model.value import ValueModel
from astrai.optimization.scheduler import SchedulerFactory
from tests.support.models import TINY_CONFIG


def make_online_model(model_config):
    """Tiny float32 LM for online-rollout tests (shared by e2e/scheduling)."""
    return AutoRegressiveLM(model_config).to(dtype=torch.float32)


def make_online_value_model(model_config):
    return ValueModel(model_config).to(dtype=torch.float32)


def make_online_optimizer(m):
    return torch.optim.AdamW(m.parameters(), lr=1e-4)


def make_online_lr_scheduler(optim, _dataset_len):
    return SchedulerFactory.create(
        "cosine", optim, warmup_steps=1, lr_decay_steps=4, min_rate=0.05
    )


def create_train_config(
    dataset,
    test_dir: str,
    device: str,
    strategy: str = "seq",
    n_epoch: int = 1,
    batch_per_device: int = 2,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    ckpt_interval: int = 5,
    random_seed: int = 42,
    **kwargs,
):
    """Factory function to create common TrainConfig for tests."""

    return TrainConfig(
        strategy=strategy,
        model=ComponentSpec(type="autoregressive_lm", params={"config": TINY_CONFIG}),
        dataset=ComponentSpec(
            type=f"{type(dataset).__module__}:{type(dataset).__qualname__}",
            params={
                **{
                    key: getattr(dataset, key)
                    for key in (
                        "length",
                        "max_length",
                        "vocab_size",
                        "with_loss_mask",
                        "stop_after",
                    )
                    if hasattr(dataset, key)
                }
            },
        ),
        optimizer=ComponentSpec(type="torch.optim:AdamW", params={"lr": 0.001}),
        scheduler=ComponentSpec(
            type="cosine",
            params={"warmup_steps": 10, "lr_decay_steps": 10, "min_rate": 0.05},
        ),
        ckpt_dir=test_dir,
        n_epoch=n_epoch,
        batch_per_device=batch_per_device,
        ckpt_interval=ckpt_interval,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        device_type=device,
        **kwargs,
    )
