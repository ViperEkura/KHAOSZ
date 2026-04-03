import pytest
import torch
from torch.utils.data import Dataset

from astrai.config import TrainConfig
from astrai.trainer.schedule import SchedulerFactory


class TrainerDataset(Dataset):
    """Base dataset for trainer tests with consistent interface."""

    def __init__(self, length=100, max_length=64, vocab_size=1000):
        self.length = length
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
            "target_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
        }


def create_train_config(
    model: torch.nn.Module,
    dataset: Dataset,
    test_dir: str,
    device: str,
    strategy: str = "seq",
    n_epoch: int = 1,
    batch_size: int = 2,
    accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    ckpt_interval: int = 5,
    random_seed: int = 42,
    **kwargs,
):
    """Factory function to create common TrainConfig for tests.

    Args:
        model: The model to train
        dataset: Training dataset
        test_dir: Checkpoint directory
        device: Device type ("cuda" or "cpu")
        strategy: Training strategy type (default: "seq")
        n_epoch: Number of epochs (default: 1)
        batch_size: Batch size (default: 2)
        accumulation_steps: Gradient accumulation steps (default: 1)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        ckpt_interval: Checkpoint save interval in iterations (default: 5)
        random_seed: Random seed for reproducibility (default: 42)
        **kwargs: Additional arguments passed to TrainConfig

    Returns:
        TrainConfig instance configured for testing
    """

    optimizer_fn = lambda m: torch.optim.AdamW(m.parameters(), lr=0.001)
    scheduler_fn = lambda optim: SchedulerFactory.create(
        optim, "cosine", warmup_steps=10, lr_decay_steps=10, min_rate=0.05
    )

    return TrainConfig(
        strategy=strategy,
        model=model,
        dataset=dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        ckpt_dir=test_dir,
        n_epoch=n_epoch,
        batch_size=batch_size,
        ckpt_interval=ckpt_interval,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        device_type=device,
        **kwargs,
    )


@pytest.fixture
def train_config_factory():
    """Fixture that provides the create_train_config factory function.

    This fixture can be used by tests to create consistent TrainConfig
    instances with sensible defaults for testing.
    """
    return create_train_config


@pytest.fixture
def trainer_dataset():
    """Fixture providing a dataset for trainer tests."""
    dataset = TrainerDataset()
    yield dataset
